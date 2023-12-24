########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, math, gc, importlib
import torch
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
import torch.nn as nn
from torch.nn import functional as F


def __nop(ob):
    return ob


def print_memory_usage():
    process = psutil.Process()
    memory_usage = process.memory_info().rss / 1024 / 1024  # in MB
    print(f"Current memory usage: {memory_usage} MB")

MyModule = nn.Module
MyFunction = __nop
if "RWKV_JIT_ON" in os.environ and os.environ["RWKV_JIT_ON"] == "1":
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method


########################################################################################################
# CUDA Kernel
########################################################################################################

from torch.utils.cpp_extension import load
import psutil

HEAD_SIZE = 64
if torch.backends.mps.is_available():
    #mps_dir is on parrallel to __file__'s director
    mps_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'mps')
    wkv5_mps = torch.utils.cpp_extension.load(
        name='RWKV5Ops',
        sources=[os.path.join(mps_dir,'RWKV5Ops_run.mm')],
        extra_cflags=['-std=c++17'],
    )
    print('compile mps kernel successfully')
elif torch.cuda.is_available():
    cuda_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'cuda')
    sources = [os.path.join(cuda_dir,'RWKV5Ops_run.cu'),os.path.join(cuda_dir,'RWKV5Ops_run.cpp')]
    wkv5_cuda = load(name="wkv5_run", sources=sources, verbose=True,
                    extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}"] )
class WKV_5(torch.autograd.Function):
    if torch.backends.mps.is_available():
        @staticmethod
        def forward(ctx, B, T, C, H,state, r, k, v, w, u):
            with torch.no_grad():
                assert r.dtype == torch.float32
                assert k.dtype == torch.float32
                assert v.dtype == torch.float32
                assert w.dtype == torch.float32
                assert u.dtype == torch.float32
                assert HEAD_SIZE == C // H
                eew = (torch.exp(-torch.exp(w))).contiguous()
                y = torch.zeros((B, T, C), device=r.device, dtype=torch.float32).contiguous() # .uniform_(-1, 1)
                wkv5_mps.wkv5_forward(B, T, C, H,state, r, k, v, eew, u, y)
                return y,state
    elif torch.cuda.is_available():
        @staticmethod
        def forward(ctx, B, T, C, H,state, r, k, v, w, u):
            with torch.no_grad():
                if state.dtype != torch.float32:
                    state = state.float()
                if r.dtype != torch.bfloat16:
                    r = r.bfloat16()
                if k.dtype != torch.bfloat16:
                    k = k.bfloat16()
                if v.dtype != torch.bfloat16:
                    v = v.bfloat16()
                if u.dtype != torch.bfloat16:
                    u = u.bfloat16()
                
                assert HEAD_SIZE == C // H
                eew = (torch.exp(-torch.exp(w.float()))).contiguous()
                y = torch.zeros((B, T, C), device=r.device, dtype=torch.bfloat16).contiguous()
                wkv5_cuda.wkv5_forward(B, T, C, H,state, r, k, v, eew, u, y)
                return y,state

def RUN_CUDA_RWKV5(B, T, C, H,state, r, k, v, w, u):
    return WKV_5.apply(B, T, C, H,state, r, k, v, w, u)

########################################################################################################

class RWKV_TimeMix_RWKV5(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size_a
        assert HEAD_SIZE == self.head_size # change HEAD_SIZE to match args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        self.head_size_divisor = args.head_size_divisor

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            # fancy time_mix
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_mix_g = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            # fancy time_decay
            decay_speed = torch.ones(args.dim_att)
            for n in range(args.dim_att):
                decay_speed[n] = -6 + 5 * (n / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(self.n_head, self.head_size))
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

            tmp = torch.zeros(args.dim_att)
            for n in range(args.dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (args.dim_att - 1))) + zigzag

            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)

        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
        self.gate = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head, args.dim_att)

    @MyFunction
    def jit_func(self, x,state_xx):
        #x is [T,C],state_xx is [C]
        #cat state_xx to x as the previous state
        sx = torch.cat((state_xx.unsqueeze(0),x[:-1,:]))
        xk = x * self.time_mix_k + sx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + sx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + sx * (1 - self.time_mix_r)
        xg = x * self.time_mix_g + sx * (1 - self.time_mix_g)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))

        return r, k, v, g

    @MyFunction
    def jit_func_2(self, x, g):
        B,T, C = x.size()
        x = x.view(B*T, C)
        
        x = self.ln_x(x / self.head_size_divisor).view(T, C)
        x = self.output(x * g)
        return x

    def forward(self, x,state_xx,state_kv):
        T, C = x.size()
        H = self.n_head
        xx = x[-1,:]
        r, k, v, g = self.jit_func(x,state_xx)
        #state_kv is (n_head,head_size,head_size)
        x,s = RUN_CUDA_RWKV5(1, T, C, H, state_kv.transpose(-1,-2).contiguous(),r, k, v, w=self.time_decay, u=self.time_faaaa)
        s = s.transpose(-1,-2)
        x = self.jit_func_2(x, g).squeeze(0)
        return x,xx,s

########################################################################################################

class RWKV_ChannelMix(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
        
        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    @MyFunction
    def forward(self,x,state_ffn):
        #x is [T,C],state _ffn is [C]
        xx = torch.cat((state_ffn.unsqueeze(0),x[:-1,:]))
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return (torch.sigmoid(self.receptance(xr)) * kv).squeeze(0), x[-1,:]

class MishGLU(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)

            x = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                x[0, 0, i] = i / args.n_embd

            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.aa = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
            self.bb = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
            self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x)
        xa = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xb = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        a = self.aa(xa)
        b = self.bb(xb)
        return self.value(a * F.mish(b))

########################################################################################################
# The RWKV Model with our blocks
########################################################################################################


class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.pre_ffn = args.pre_ffn
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)

        self.att = RWKV_TimeMix_RWKV5(args, layer_id)

        self.ffn = RWKV_ChannelMix(args, layer_id)
        
    def forward(self, x,state):
        if self.layer_id == 0:
            x = self.ln0(x)

        x_,x_x,state_kv = self.att(self.ln1(x),state[0],state[1])
        x = x +x_
        x_,state_ffn = self.ffn(self.ln2(x),state[2])
        x = x + x_
        return x,x_x,state_kv,state_ffn


class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)


class RWKV(MyModule):
    def __init__(self, args,device='cuda'):
        super().__init__()
        self.args = args
        if not hasattr(args, 'dim_att'):
            args.dim_att = args.n_embd
        if not hasattr(args, 'dim_ffn'):
            args.dim_ffn = args.n_embd * 4
        assert args.n_embd % 32 == 0
        assert args.dim_att % 32 == 0
        assert args.dim_ffn % 32 == 0
        self.device = device

        self.emb = nn.Embedding(args.vocab_size, args.n_embd)
        self.n_layer = args.n_layer
        self.n_embd = args.n_embd
        self.n_att = args.dim_att
        self.head_size_a = args.head_size_a
        self.n_head = args.dim_att // args.head_size_a
        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])

        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

    def eval(self):
        super().eval()
        self.emb = self.emb.to('cpu')
        self.head = self.head.to('cpu')
        gc.collect()



    def forward(self, idx,state=None):
        with torch.no_grad():
            if isinstance(idx,torch.Tensor):
                idx_shape = idx.shape
                if len(idx_shape) == 2:
                    idx = idx.squeeze(0)
                idx = idx.to('cpu')
            elif isinstance(idx,list):
                idx = torch.tensor(idx,device='cpu',requires_grad=False)
            if state is None:
                state = [None] * self.n_layer * 3
                for i in range(self.n_layer): # state: 0=att_xx 1=att_kv 2=ffn_xx
                    state[i*3+0] = torch.zeros(self.n_embd, dtype=torch.float32, requires_grad=False, device=self.device).contiguous()
                    state[i*3+1] = torch.zeros((self.n_head, self.head_size_a, self.head_size_a), dtype=torch.float32, requires_grad=False, device=self.device).contiguous()
                    state[i*3+2] = torch.zeros(self.n_embd, dtype=torch.float32, requires_grad=False, device=self.device).contiguous()

            x = self.emb(idx).to(self.device)

            for i, block in enumerate(self.blocks):
                x,x_x,state_kv,state_ffn = block(x,state[i*3:i*3+3])
                state[i*3+0] = x_x
                state[i*3+1] = state_kv
                state[i*3+2] = state_ffn

            x = self.ln_out(x)
            
            x = self.head(x.to('cpu').bfloat16())

            return x[-1,:],state

import torch.utils.cpp_extension
from torch.utils.cpp_extension import load
import os
HEAD_SIZE = 64
cuda_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'cuda')
sources = [os.path.join(cuda_dir,'RWKV5Ops_run.cpp'),os.path.join(cuda_dir,'RWKV5Ops_run.cu')]
wkv5_cuda = load(name="wkv5_run", sources=sources, verbose=True,
                    extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}"] )

print(wkv5_cuda)

B,T,C,H = 2,3,1024,16
device = 'cuda'
r = torch.randn((B, T, C), device=device, requires_grad=False, dtype=torch.bfloat16).contiguous()
k = torch.randn((B, T, C), device=device, requires_grad=False, dtype=torch.bfloat16).contiguous()
v = torch.randn((B, T, C), device=device, requires_grad=False, dtype=torch.bfloat16).contiguous()

w = torch.randn((B, C), device=device, requires_grad=False, dtype=torch.float).contiguous()
ew = (-torch.exp(w.float())).contiguous()
eew = (torch.exp(ew)).contiguous()
u = torch.randn((B, C), device=device, requires_grad=False, dtype=torch.bfloat16).contiguous()
y = torch.empty((B, T, C), device=device, requires_grad=False, dtype=torch.bfloat16).contiguous()
print(y)
print('-----------------------')
state_orig = torch.zeros((H,HEAD_SIZE,HEAD_SIZE),device=device,requires_grad=True,dtype=torch.float32)
state = state_orig.unsqueeze(0).repeat(B, 1, 1, 1).contiguous()
wkv5_cuda.forward(B,T,C,H,state.transpose(-1,-2),r,k,v,eew,u,y)
print(y)
print('----------Y-------------')
print(state)
print('--------STATE---------------')
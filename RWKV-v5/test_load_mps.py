import types, gc, os, time, re
os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '0'
os.environ['RWKV_HEAD_SIZE_A'] = '64'
args = types.SimpleNamespace()
import torch
ckpt_file = '/Users/yueyulin/Downloads/RWKV-5-World-3B-v2-20231113-ctx4096.pth'
def load_ckpt_and_parse_args(ckpt_file, args):
    with torch.no_grad():
        w = torch.load(ckpt_file, map_location='cpu') # load model to CPU first
        gc.collect()
        n_embd = w['emb.weight'].shape[1]
        vocab_size = w['emb.weight'].shape[0]
        dim_att = w['blocks.0.att.key.weight'].shape[0] # note: transposed matrix
        dim_ffn = w['blocks.0.ffn.key.weight'].shape[0] # note: transposed matrix
        n_layer = 0
        keys = list(w.keys())
        version = 4
        for x in keys:
            layer_id = int(x.split('.')[1]) if ('blocks.' in x) else 0
            n_layer = max(n_layer, layer_id+1)
            if 'ln_x' in x:
                version = max(5, version)
            if 'gate.weight' in x:
                version = max(5.1, version)
            if int(version) == 5 and 'att.time_decay' in x:
                n_head = w[x].shape[0]
                if len(w[x].shape) > 1:
                    if w[x].shape[1] > 1:
                        version = max(5.2, version)
        head_size_a = dim_att // n_head
        args.n_embd = n_embd
        args.dim_att = dim_att
        args.dim_ffn = dim_ffn
        args.n_layer = n_layer
        args.version = version
        args.head_size_a = head_size_a
        args.vocab_size = vocab_size
        return w

w = load_ckpt_and_parse_args(ckpt_file, args)
args.head_qk = 0
args.dropout = 0
args.my_pos_emb = 0
args.pre_ffn = 0
args.head_size_divisor = 8
args.ctx_len = 4096
args.grad_cp = 0
print(args)
from src.model import RWKV
model = RWKV(args)
# print(model.state_dict().keys())

model.load_state_dict(w,strict=True)
model = model.to('mps')
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
tokenizer = TRIE_TOKENIZER(os.path.dirname(os.path.abspath(__file__)) + '/rwkv_vocab_v20230424.txt' )
ids = tokenizer.encode("很久很久以前，有一个")
print(ids)
model.eval()
ids = torch.tensor([ids],dtype=torch.int,device='mps');
print(ids)
print(ids.size())
x = model.forward(ids)
print(x.shape)
print(x)  
ids_out = torch.argmax(x,2).cpu().numpy()[0].tolist()
print(ids_out)
print(tokenizer.decode(ids_out))
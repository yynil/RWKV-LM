import os
'''
export RWKV_JIT_ON=1
export RWKV_T_MAX=1024
export RWKV_FLOAT_MODE=fp32
'''
os.environ['RWKV_JIT_ON'] = '1'
os.environ['RWKV_T_MAX'] = '1024'
os.environ['RWKV_FLOAT_MODE'] = 'fp32'
import torch
from argparse import Namespace
args = Namespace()
args.head_size=0
args.dim_att = 1024
args.layer_id = 0
args.ctx_len = 1024
args.n_embd = 1024
args.dim_ffn = args.n_embd*4
args.n_layer = 12
from src.model import RWKV_TimeMix
time_mix = RWKV_TimeMix(args,0).to('mps')
B,T,C = 32,1024,1024
x = torch.randn((B,T,C),dtype=torch.float32,device='mps')
rwkv = time_mix(x)
print(rwkv)

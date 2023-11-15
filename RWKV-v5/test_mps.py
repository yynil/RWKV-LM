import src.model
from src.model import WKV_5
B = 32
T = 128 
C = 1024
H = 16
import torch
r = torch.rand((B,T,C),device='mps',dtype=torch.float32)
k = torch.rand((B,T,C),device='mps',dtype=torch.float32)
v = torch.rand((B,T,C),device='mps',dtype=torch.float32)
w = torch.rand(C,device='mps',dtype=torch.float32)
u = torch.rand(C,device='mps',dtype=torch.float32)
from src.model import RUN_CUDA_RWKV5
y = RUN_CUDA_RWKV5(B,T,C,H,r,k,v,w,u)
print(y)
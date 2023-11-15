import torch.utils.cpp_extension

compiled_lib = torch.utils.cpp_extension.load(
    name='RWKV5Ops',
    sources=['RWKV5Ops.mm'],
    extra_cflags=['-std=c++17'],
   )

print(compiled_lib)

B = 32
T = 128 
C = 1024
H = 16
import torch
r = torch.rand((B,T,C),device='mps',dtype=torch.float32)
print(r)
k = torch.rand((B,T,C),device='mps',dtype=torch.float32)
print(k)
v = torch.rand((B,T,C),device='mps',dtype=torch.float32)
print(v)
w = torch.rand(C,device='mps',dtype=torch.float32)
print(w)
u = torch.rand(C,device='mps',dtype=torch.float32)
print(u)
y = torch.zeros_like(v,device='mps',dtype=torch.float32,memory_format=torch.contiguous_format)
print(y)
r = r.contiguous()
k = k.contiguous()
v = v.contiguous()
w = w.contiguous()
u = u.contiguous()
import datetime
start = datetime.datetime.now()
ew = -torch.exp(w)
eew = torch.exp(ew)
compiled_lib.wkv5_forward(B,T,C,H,r,k,v,eew,u,y)
end = datetime.datetime.now()
print(y)
print(f'forward time: {(end-start).total_seconds()}')
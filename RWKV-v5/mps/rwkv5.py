import torch.utils.cpp_extension
import os
directory = os.path.dirname(os.path.abspath(__file__))
file_name = os.path.join(directory, 'RWKV5Ops.mm')
compiled_lib = torch.utils.cpp_extension.load(
    name='RWKV5Ops',
    sources=[file_name],
    extra_cflags=['-std=c++17'],
   )

print(compiled_lib)

B = 1
T = 128 
C = 1024
H = 16
import torch
r = torch.rand((B,T,C),device='mps',dtype=torch.float32,requires_grad=True)
print(r)
k = torch.rand((B,T,C),device='mps',dtype=torch.float32,requires_grad=True)
print(k)
v = torch.rand((B,T,C),device='mps',dtype=torch.float32,requires_grad=True)
print(v)
w = torch.rand(C,device='mps',dtype=torch.float32,requires_grad=True)
print(w)
u = torch.rand(C,device='mps',dtype=torch.float32,requires_grad=True)
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
gy = torch.ones_like(y,device='mps',dtype=torch.float32,memory_format=torch.contiguous_format)
gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
gw = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
compiled_lib.wkv5_backward(B, T, C, H, r, k, v, eew, ew, u, gy, gr, gk, gv, gw, gu)
gw = torch.sum(gw, 0).view(H, C//H)
gu = torch.sum(gu, 0).view(H, C//H)
print(gw)
print(gu)
print(gr)
print(gk)
print(gv)
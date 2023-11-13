import torch.utils.cpp_extension

compiled_lib = torch.utils.cpp_extension.load(
    name='RWKVOps',
    sources=['RWKVOps.mm'],
    extra_cflags=['-std=c++17'],
   )

print(compiled_lib)

B = 32
T = 128 
C = 1024
import torch
w = torch.randn(C, device='mps',dtype=torch.float32)
print(w)
u = torch.randn(C,device='mps',dtype=torch.float32)
print(u)
k = torch.randn((B,T,C),device='mps',dtype=torch.float32)
print(k)
v = torch.randn((B,T,C),device='mps',dtype=torch.float32)
print(v)
y = torch.zeros_like(v,device='mps',dtype=torch.float32,memory_format=torch.contiguous_format)
print(y)
w = w.contiguous()
u = u.contiguous()
k = k.contiguous()
v = v.contiguous()
import datetime
start = datetime.datetime.now()
compiled_lib.wkv_forward(B,T,C,w,u,k,v,y)
end = datetime.datetime.now()
print(y)
print(f'forward time: {(end-start).total_seconds()}')
y = y.contiguous()
dy = torch.randn((B,T,C),device='mps',dtype=torch.float32)
dy = dy.contiguous()
# print(dy)
dw = torch.empty((B, C),device='mps',dtype=torch.float32,memory_format=torch.contiguous_format)
# print(dw)
du = torch.empty((B, C),device='mps',dtype=torch.float32,memory_format=torch.contiguous_format)
# print(du)
dk = torch.empty((B, T, C),device='mps',dtype=torch.float32,memory_format=torch.contiguous_format)
# print(dk)
dv = torch.empty((B, T, C),device='mps',dtype=torch.float32,memory_format=torch.contiguous_format)
# print(dv)
start = datetime.datetime.now()
compiled_lib.wkv_backward(B,T,C,
                          w,u,k,v,y,dy,
                          dw,du,dk,dv)
end = datetime.datetime.now()
print(dw)
print(du)
print(dk)
print(dv)
print(f'backward time: {(end-start).total_seconds()}')
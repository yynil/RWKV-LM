import torch.utils.cpp_extension
import os
directory = os.path.dirname(os.path.abspath(__file__))
file_name = os.path.join(directory, 'RWKV5Ops.mm')
wkv5_mps = torch.utils.cpp_extension.load(
    name='RWKV5Ops',
    sources=[file_name],
    extra_cflags=['-std=c++17'],
   )

print(wkv5_mps)

data_dir = os.path.dirname(directory)
mps_file = os.path.join(data_dir,'mps.json')
print(data_dir,mps_file)

import json
data = json.load(open(mps_file,'r'))
print(data.keys())

import torch
B,T,C,H = data['B'],data['T'],data['C'],data['H']
print(B,T,C,H)
r = torch.tensor(data['r'],dtype=torch.float32,device='mps')
k = torch.tensor(data['k'],dtype=torch.float32,device='mps')
v = torch.tensor(data['v'],dtype=torch.float32,device='mps')
print(r)
print(k)
print(v)
w = torch.tensor(data['w'],dtype=torch.float32,device='mps')
eew = torch.tensor(data['eew'],dtype=torch.float32,device='mps')
ew = torch.tensor(data['ew'],dtype=torch.float32,device='mps')

ew_cal = (-torch.exp(w.float())).contiguous()
eew_cal = (torch.exp(ew)).contiguous()
print(f'Mean eew_eew_cal {torch.mean(torch.abs(eew - eew_cal))}')
print(f'Mean ew_ew_cal {torch.mean(torch.abs(ew - ew_cal))}')

u = torch.tensor(data['u'],dtype=torch.float32,device='mps')
y = torch.tensor(data['y'],dtype=torch.float32,device='mps')
yt = torch.empty((B, T, C), device=r.device, dtype=torch.float32, memory_format=torch.contiguous_format)

wkv5_mps.wkv5_forward(B,T,C,H,r,k,v,eew,u,yt)
print(y)
print(yt)
yt = yt.reshape((B*T*C))
print(yt)
mean = torch.mean(torch.abs(y - yt))
print(f"forward Mean: {mean}")
gy = torch.tensor(data['gy'],dtype=torch.float32,device='mps')
print(gy)
grt = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
gkt = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
gvt = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
gwt = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
gut = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format) # .uniform_(-1, 1)

wkv5_mps.wkv5_backward(B,T,C,H,r,k,v,eew_cal,ew_cal,u,gy,grt,gkt,gvt,gwt,gut)
# print(grt)
# print(gkt)
# print(gvt)
# print(gvt)
# print(gwt)
# print(gut)


gr = torch.tensor(data['gr'],dtype=torch.float32,device='mps').reshape((B,T,C))
gk = torch.tensor(data['gk'],dtype=torch.float32,device='mps').reshape((B,T,C))
gv = torch.tensor(data['gv'],dtype=torch.float32,device='mps').reshape((B,T,C))
gw = torch.tensor(data['gw'],dtype=torch.float32,device='mps').reshape((B,C))
gu = torch.tensor(data['gu'],dtype=torch.float32,device='mps').reshape((B,C))
print(gr)
print(gk)
print(gv)
print(gw)
print(gu)

print(f'Mean gr {torch.mean(torch.abs(gr - grt))}')
print(f'Mean gk {torch.mean(torch.abs(gk - gkt))}')
print(f'Mean gv {torch.mean(torch.abs(gv - gvt))}')
print(f'Mean gw {torch.mean(torch.abs(gw - gwt))}')
print(f'Mean gu {torch.mean(torch.abs(gu - gut))}')
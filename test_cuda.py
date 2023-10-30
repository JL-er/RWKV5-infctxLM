import torch
from torch.utils.cpp_extension import load



wkv5_cuda = load(name="wkv51", sources=["cuda/wkv5_op_state.cpp", f"cuda/wkv5_cuda_state.cu"],
                    verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_=8"])

torch.manual_seed(41)


B = 1
T = 8
C = 8
H = 1

# 创建测试输入张量
r = torch.randn(B, T, C, device='cuda', dtype=torch.bfloat16).contiguous()
k = torch.randn(B, T, C,device='cuda', dtype=torch.bfloat16).contiguous()
v = torch.randn(B, T, C,device='cuda', dtype=torch.bfloat16).contiguous()
w = torch.randn(B, C,device='cuda', dtype=torch.float32).contiguous()
u = torch.randn(B, C,device='cuda', dtype=torch.bfloat16).contiguous()
y = torch.zeros(B, T, C, device='cuda', dtype=torch.bfloat16).contiguous()
ls = torch.zeros((B, H, C//H, C//H), device='cuda', dtype=torch.float).contiguous()
ns = torch.zeros((B, H, C//H, C//H), device='cuda', dtype=torch.float).contiguous()


r1 = r[:, :-4, :].clone()
k1 = k[:, :-4, :].clone()
v1 = v[:, :-4, :].clone()
w1 = w.clone()
u1 = u.clone()
y1 = torch.zeros(B, 4, C, device='cuda', dtype=torch.bfloat16).contiguous()
ls1 = ls.clone()
ns1 = ns.clone()

r2 = r[:, 4:, :].clone()
k2 = k[:, 4:, :].clone()
v2 = v[:, 4:, :].clone()
w2 = w.clone()
u2 = u.clone()
y2 = torch.zeros(B, 4, C, device='cuda', dtype=torch.bfloat16).contiguous()
ls2 = ls.clone()
ns2 = ns.clone()

# print(ls.shape)
# # 调用前向传播函数
# print(ns)

# k1 = k1[:,:-2,:]
# print(k)
# print(k1)
wkv5_cuda.forward(B, T, C, H, r, k, v, w, u, y, ls, ns)
# print(k)
# print(k1)
# print(ns)
print(y)

wkv5_cuda.forward(B, 4, C, H, r1, k1, v1, w, u, y1, ls1, ns1)
#print(ns1)
ns2 = torch.zeros((B, H, C//H, C//H), device='cuda', dtype=torch.float32).contiguous()

wkv5_cuda.forward(B, 4, C, H, r2, k2, v2, w, u, y2, ns1, ns2)
#print(ns2)

print(y2)
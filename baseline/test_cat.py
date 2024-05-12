""" 这个代码证明了concat操作依然是需要空间的 """
import torch

is_cat = True

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

x1 = torch.randn(8, 32, requires_grad=True).cuda()
x2 = torch.randn(8, 32, requires_grad=True).cuda()
if is_cat:
    x2 = torch.cat([x1, x2], dim=-1)

max_tensor_memory = torch.cuda.max_memory_allocated()

print(f'{max_tensor_memory} bytes')


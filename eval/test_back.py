""" 本代码尝试在OOM后，释放计算图 """

import torch
from numba import cuda
import gc


a = torch.tensor([1,2,3.], requires_grad=True).cuda()
b = torch.tensor([1,2,3.], requires_grad=True).cuda()
print("Current GPU memory allocated:", torch.cuda.memory_allocated())
a2 = a * 2
list.append(a2)
b2 = torch.sum(b)
list.append(b2)
print("Current GPU memory allocated:", torch.cuda.memory_allocated())
b2.backward()
# device = cuda.get_current_device()
# device.reset()

# torch.cuda.empty_cache()
torch.cuda.empty_cache()
print("Current GPU memory allocated:", torch.cuda.memory_allocated())

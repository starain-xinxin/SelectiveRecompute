import torch.nn as nn
import torch

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        for i, l in enumerate(self.linears):
            x = self.linears[i // 2](x) + l(x)
        return x

model = MyModule()
a = torch.tensor([1,2,3,4,5,6,7,8,9,0.])
b = model(a)
print()
import torch
import torch.nn as nn

output = [
    [0.9, 0.05, 0.05],
    [0.8, 0.05, 0.15]
]
output = torch.tensor(output, requires_grad=True)

label = [0,1]
label = torch.tensor(label)

loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(output, label)
print(loss)

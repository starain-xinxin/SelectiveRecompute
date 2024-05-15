import torch
import Checkmodel as Cm

# 实验变量
device = 'cuda:0'
batch_size = 30

seq_len = 1024
d_model = 512
dim_k = 512
dim_v = 512
num_encoder = 10
num_head = 4
hidden_dim = 2024
activation_func = 'ReLU'
num_class = 20

# --------------- 0.监控代码 --------------- #

# --------------- 1.构建模型 --------------- #
model = Cm.CheckBert(seq_len, d_model, dim_k, dim_v,  num_encoder, num_head, hidden_dim, activation_func, num_class).to(device)

# --------------- 2.构建数据 --------------- #
data = torch.randn((batch_size, seq_len, d_model)).to(device)
label = random_labels = torch.randint(low=0, high=num_class, size=(batch_size,)).to(device)

# --------------- 3.计算输出 --------------- #
out = model(data)
loss = model.BertLoss(out, label)
loss.backward()

print(out.size())

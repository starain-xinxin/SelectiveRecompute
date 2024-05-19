import torch
import Checkmodel as Cm
import Pretrain

# 实验变量
device = 'cuda:0'
batch_size = 30

seq_len = 1024
d_model = 512
dim_k = 512
dim_v = 512
num_encoder = 2
num_head = 1
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

# --------------- 4.构建 check model 森林 --------------- #
# (1) 构建森林
Check_forest = Pretrain.CheckForest(model)
# (2) 计算 profit 和 cost
Check_forest.make_profit_cost()
# (3) 测试 make_forest_type_check
Pretrain.make_forest_type_check(Check_forest, gradient_checkpoint=False, model_type=Cm.MultiHeadQKV)
Pretrain.make_forest_type_check(Check_forest, gradient_checkpoint=False, model_type='All')
Pretrain.make_forest_type_check(Check_forest, gradient_checkpoint=True, model_type='All')

print(out.size())
print(f'{type(model).__name__}')
print(model._modules['encoders']._modules['0']._modules['MultiAttention']._compute_profit())

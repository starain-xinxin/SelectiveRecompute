from torch.utils.data import DataLoader
import Checkmodel as Cm
import Pretrain

# 实验变量
# device = 'cuda:0'
# batch_size = 10
#
# seq_len = 2024
# d_model = 512
# num_encoder = 2
# num_head = 16
# dim_k = int(512 / num_head)
# dim_v = int(512 / num_head)
# hidden_dim = 2024
# activation_func = 'ReLU'
# num_class = 20

device = 'cuda:0'
batch_size = 2

seq_len = 8 * 16
d_model = 8 * 16
num_encoder = 1
num_head = 8
dim_k = int(d_model / num_head)
dim_v = int(d_model / num_head)
hidden_dim = 2 * 8 * 16
activation_func = 'ReLU'
num_class = 8

# --------------- 0.监控代码 --------------- #

# --------------- 1.构建模型 --------------- #
model = Cm.CheckBert(seq_len, d_model, dim_k, dim_v,  num_encoder, num_head, hidden_dim, activation_func, num_class)

# --------------- 2.构建 dataloader --------------- #
data_maker = Pretrain.DataMaker(batch_size, seq_len, d_model, num_class)
dataloader = DataLoader(data_maker, batch_size=batch_size, shuffle=True)

# --------------- 3.launch --------------- #
Pretrain.Pre_launch(model, dataloader, model.BertLoss, device=device)


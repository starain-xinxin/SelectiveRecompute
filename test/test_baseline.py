from torch.utils.data import DataLoader
import Checkmodel as Cm
import Pretrain
import logging
import utils
import torch.optim as optim
from tqdm import tqdm
logging.basicConfig(level=logging.INFO)

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

# max_GPU_memory = 4
# device0 = 'cuda:0'
# device1 = 'cuda:1'
# device2 = 'cuda:2'
# iters = 5
#
# batch_size = 10
# seq_len = 1024
# d_model = 512
# num_encoder = 8
# num_head = 8
# dim_k = int(d_model / num_head)
# dim_v = int(d_model / num_head)
# hidden_dim = 2 * d_model
# activation_func = 'ReLU'
# num_class = 8

# max_GPU_memory = 80
# device0 = 'cuda:0'
# device1 = 'cuda:1'
# device2 = 'cuda:2'
# iters = 20

# batch_size = 5
# seq_len = 1024
# d_model = 512
# num_encoder = 16
# num_head = 8
# dim_k = int(d_model / 4)
# dim_v = int(d_model / 4)
# hidden_dim = 2 * d_model
# activation_func = 'ReLU'
# num_class = 10

max_GPU_memory = 80
device0 = 'cuda:0'
device1 = 'cuda:1'
device2 = 'cuda:2'
iters = 20

batch_size = 25
seq_len = 1024
d_model = 1024
num_encoder = 40
num_head = 20
dim_k = int(d_model / 4)
dim_v = int(d_model / 4)
hidden_dim = 2 * d_model
activation_func = 'ReLU'
num_class = 10

# --------------- 0.监控代码 --------------- #

# --------------- 1.构建模型,优化器 --------------- #
model = Cm.CheckBert(seq_len, d_model, dim_k, dim_v,  num_encoder, num_head, hidden_dim, activation_func, num_class)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --------------- 2.构建 dataloader --------------- #
data_maker = Pretrain.DataMaker(batch_size, seq_len, d_model, num_class, iters=iters)
dataloader = DataLoader(data_maker, batch_size=batch_size, shuffle=True)

# --------------- 3.baseline --------------- #
# (1) 构建森林
Check_forest = Pretrain.CheckForest(model)
# (2) baseline
Pretrain.make_forest_type_check(Check_forest, gradient_checkpoint=False, model_type='All')
Pretrain.make_forest_type_check(Check_forest, gradient_checkpoint=True, model_type=Cm.MultiHeadQKV)

# --------------- 4.train ---------------- #
logging.info(f'---------------- Start Train -----------------------')

mem_monitor = utils.MemoryMonitor(folder='/home/yuanxinyu/SelectiveRecompute/data/test', name='baseline',
                                  device=device1, is_snapshot=True)
mem_monitor.start()

model.to(device1)

timer = utils.Timer('baseline')
timer.start()

## train ##
for batch_idx, (inputs, targets) in tqdm(enumerate(dataloader)):
    inputs = inputs.to(device1)
    targets = targets.to(device1)
    out = model(inputs)
    optimizer.zero_grad()
    loss = model.BertLoss(out, targets)
    loss.backward()
    optimizer.step()
## train ##

timer.end()
logging.info(f'总时间{timer.runtime()}')
mem_monitor.end()
max_tensor_memory = mem_monitor.max_tensor_memory
logging.info(f'selective策略下，最大张量占用：{max_tensor_memory / 1024}GB')
max_torch_memory = mem_monitor.max_torch_cache
logging.info(f'selective策略下，最大张量占用：{max_torch_memory / 1024}GB')



from torch.utils.data import DataLoader
import Checkmodel as Cm
import Pretrain
import logging
import utils
import torch.optim as optim
from tqdm import tqdm
logging.basicConfig(level=logging.INFO)

def calculate_model_size(model):
    """计算模型的参数数量，并返回以合适单位（K、M、B）表示的参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    if total_params >= 10**9:
        # 以B为单位
        unit = 'B'
        size = total_params / 10**9
    elif total_params >= 10**6:
        # 以M为单位
        unit = 'M'
        size = total_params / 10**6
    elif total_params >= 10**3:
        # 以K为单位
        unit = 'K'
        size = total_params / 10**3
    else:
        # 参数数量太小，直接使用
        unit = ''
        size = total_params
    return size, unit

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

max_GPU_memory = 80
device0 = 'cuda:0'
device1 = 'cuda:1'
device2 = 'cuda:2'
iters = 20

batch_size = 27
seq_len = 2048
d_model = 512
num_encoder = 16
num_head = 8
dim_k = int(d_model / num_head)
dim_v = int(d_model / num_head)
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

# --------------- 3.launch --------------- #
# Pretrain.Pre_launch(model, dataloader, model.BertLoss, device=device1, tolerance=0.90, max_GPU_memory=max_GPU_memory)

# --------------- 4.train ---------------- #
logging.info(f'---------------- Start Train -----------------------')

mem_monitor = utils.MemoryMonitor(folder='/home/yuanxinyu/SelectiveRecompute/data/test', name='OOM',
                                  device=device0, is_snapshot=True)
mem_monitor.start()

model.to(device0)

timer = utils.Timer('OOM')
timer.start()

## train ##
try:
    for batch_idx, (inputs, targets) in tqdm(enumerate(dataloader)):
        inputs = inputs.to(device0)
        targets = targets.to(device0)
        out = model(inputs)
        optimizer.zero_grad()
        loss = model.BertLoss(out, targets)
        loss.backward()
        optimizer.step()
except RuntimeError as e:
    print(f'错误信息{e}')
    timer.end()
    logging.info(f'总时间{timer.runtime()}')
    mem_monitor.end()
    max_tensor_memory = mem_monitor.max_tensor_memory
    logging.info(f'无重计算最大张量占用：{max_tensor_memory / 1024}GB')
    size, unit = calculate_model_size(model)
    logging.info(f'模型参数量：{size}{unit}')
    max_torch_memory = mem_monitor.max_torch_cache
    logging.info(f'无重计算最大显存占用：{max_torch_memory / 1024}GB')

## train ##

# timer.end()
# logging.info(f'总时间{timer.runtime()}')
# mem_monitor.end()
# max_tensor_memory = mem_monitor.max_tensor_memory
# logging.info(f'无重计算最大张量占用：{max_tensor_memory / 1024}GB')
# size, unit = calculate_model_size(model)
# logging.info(f'模型参数量：{size}{unit}')
# max_torch_memory = mem_monitor.max_torch_cache
# logging.info(f'无重计算最大显存占用：{max_torch_memory / 1024}GB')






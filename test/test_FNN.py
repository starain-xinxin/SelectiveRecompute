import Checkmodel as Cm
import torch
import utils
import logging
from tqdm import tqdm
import numpy as np
logging.basicConfig(level=logging.INFO)

# 定义数据常量
gradient_checkpoint = True

batch_size = 10
seq_len = 128
d_model = 24

hidden_dim = 512

Iters = 2

filename = 'test_FNN_backward_' + f'{gradient_checkpoint}'

# 开启内存监控
mem_monitor = utils.MemoryMonitor(folder='/home/yuanxinyu/SelectiveRecompute/data/test', name=filename)
mem_monitor.start()

# 构建模型,优化器
class Model(Cm.CheckModule):
    def __init__(self, d_model, hidden_dim, gradient_checkpoint):
        super().__init__()
        self.fnn1 = Cm.FNN(d_model, hidden_dim, gradient_checkpoint=gradient_checkpoint)
        self.fnn2 = Cm.FNN(d_model, hidden_dim, gradient_checkpoint=gradient_checkpoint)
    def forward(self, x):
        x = self.fnn1(x)
        x = self.fnn2(x)
        return x
model = Model(d_model, hidden_dim, gradient_checkpoint).cuda()
optimizer = torch.optim.Adam(model.parameters())

data = torch.randn(size=(batch_size, seq_len, d_model), requires_grad=True).cuda()
output = model(data)
label = torch.randn(size=output.size()).cuda()

# 开启计时器
timer = utils.Timer(filename)
timer.start()

# 开始训练
for i in tqdm(range(Iters)):
    # 前向计算
    output = model(data)

    label = torch.randn(size=output.size()).cuda()
    loss = torch.nn.functional.cross_entropy(output, label)



    # 反向计算
    loss.backward()

# 反向向监视器关闭
mem_monitor.end()



# 关闭内存监控,计时器
timer.end()
# mem_monitor.end()

# 记录数据
time = timer.run_time
max_torch_cache = mem_monitor.max_torch_cache
max_tensor_memory = mem_monitor.max_tensor_memory
logging.info(f'计算收益：{model.fnn1._compute_profit()}')
logging.info(f'缓存代价：{model.fnn1._memory_cost() / (1024 ** 2)}')
logging.info(f'价值比：{model.fnn1._compute_profit() / model.fnn1._memory_cost() * 4}')

# 单个模型占用的
total_paras = 0
for param in model.fnn1.parameters():
    total_paras += np.prod(param.size()) * param.element_size()
logging.info(f'{total_paras / (1024 ** 2)}')






import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import time
from tqdm import tqdm

class FNN(nn.Module):
    def __init__(self, save_mem=True, d_model = 512, hidden = 50000):
        super().__init__()
        self.save_mem = save_mem
        self.l1 = nn.Linear(d_model, hidden)
        self.s = nn.Softmax(dim=-1)
        # self.l2 = nn.Linear(hidden, hidden)
        # self.s2 = nn.Softmax(dim=-1)
        self.l3 = nn.Linear(hidden, d_model)
        self.test = nn.Linear(d_model, d_model)
    def forward(self, x):
        x = self.l1(x)
        x = self.s(x)
        # x = self.l2(x)
        # x = self.s2(x)
        x = self.l3(x)
        return x

class Net(nn.Module):
    def __init__(self, save_mem):
        super().__init__()
        self.save_mem = save_mem
        self.fnn1 = FNN(hidden=6000)
        self.fnn2 = FNN(hidden=8000)
        self.fnn3 = FNN(hidden=10000)
        self.fnn4 = FNN(hidden=8000)
        self.fnn5 = FNN(hidden=6000)    # 这里实验可以证明FNN的隐藏层越少，重计算的收益越高
    def forward(self, x):
        if self.save_mem:
            x = self.fnn1(x)
            x = cp.checkpoint(self.fnn2, x)
            x = cp.checkpoint(self.fnn3, x)
            x = cp.checkpoint(self.fnn4, x)
            x = cp.checkpoint(self.fnn5, x)
        else:
            x = self.fnn1(x)
            x = self.fnn2(x)
            x = self.fnn3(x)
            x = self.fnn4(x)
            x = self.fnn5(x)
        return x
    
    def hh(self, x):
                x = self.fnn2(x)
                x = self.fnn3(x)
                x = self.fnn4(x)
    
def train(save_mem, iters = 20, d_model = 512):
    net = Net(save_mem).cuda()
    net.train()
    loss_func = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(net.parameters())

    for _ in tqdm(range(iters)):
        data = torch.randn(size=(50, 1024, d_model)).cuda()
        label = torch.randn(size=(50, 1024, d_model)).cuda()
        output = net(data)
        # loss = loss_func(output, label)
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

if __name__ == "__main__":

    # 开启记录，并设置最多记录100000个数据点
    torch.cuda.memory._record_memory_history(max_entries=100000)

    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    F_start = time.time()
    train(True)
    F_time = time.time() - F_start
    F_a = torch.cuda.max_memory_allocated() / 1024 / 1024
    F_b = torch.cuda.max_memory_reserved() / 1024 / 1024
    print(F_a)
    print(F_b)

    # 保存数据
    torch.cuda.memory._dump_snapshot('./data/baseline/fnn-baseline-False.pickle')

    # 停掉记录，关闭snapshot
    torch.cuda.memory._record_memory_history(enabled=None)

    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    T_start = time.time()
    # train(True)
    T_time = time.time() - T_start
    T_a = torch.cuda.max_memory_allocated() / 1024 / 1024
    T_b = torch.cuda.max_memory_reserved() / 1024 / 1024
    print(T_a)
    print(T_b)

    # print(f'内存缩减：{(F_a-T_a) / F_a * 100}% 和 {(F_b-T_b) / F_b * 100}%')
    # print(f'时间增加：{(T_time - F_time) / F_time * 100}%')
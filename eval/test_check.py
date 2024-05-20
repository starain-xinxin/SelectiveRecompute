import torch
import torch.nn as nn
import numpy as np
import my_check as cp
import random
import time

# 开启记录，并设置最多记录100000个数据点
torch.cuda.memory._record_memory_history(max_entries=100000)

start_time = time.time()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

# 设置随机数种子
setup_seed(20)

device = 'cuda:0'
# device = 'cpu'

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
x = torch.Tensor(x).float().to(device)
y = np.array([1, 0, 0, 1])
y = torch.Tensor(y).long().to(device)


class MyNet(nn.Module):
    def __init__(self, save_memory=False):
        super(MyNet, self).__init__()

        self.linear1 = nn.Linear(2, 50000)
        self.linear2 = nn.Linear(50000, 30000)
        self.linear3 = nn.Linear(30000,30000)
        self.linear4 = nn.Linear(30000,3000)

        self.linear = nn.Linear(3000, 2)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)

        self.save_memory = save_memory

    def forward2(self, x):
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        x = self.linear(x)
        return x

    def forward(self, x):
        if self.save_memory:
            x = self.linear1(x)
            x = self.relu(x)
            x = cp.checkpoint(self.forward2, x)
        else:
            x = self.linear1(x)
            x = self.relu(x)
            x = self.linear2(x)
            x = self.linear3(x)
            x = self.linear4(x)
            x = self.linear(x)

        return x


torch.cuda.empty_cache()
torch.cuda.reset_max_memory_allocated()

net = MyNet(save_memory=True).to(device)
# train() enables some modules like dropout, and eval() does the opposit
net.train()

# set the optimizer where lr is the learning-rate
optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
loss_func = nn.CrossEntropyLoss().to(device)

for epoch in range(500):
    if epoch % 500 == 0:
        # call eval() and evaluate the model on the validation set
        # when calculate the loss value or evaluate the model on the validation set,
        # it's suggested to use "with torch.no_grad()" to pretrained the memory. Here I didn't use it.
        net.eval()
        out = net(x)
        loss = loss_func(out, y)
        print(loss.detach())
        # call train() and train the model on the training set
        net.train()

    out = net(x)
    loss = loss_func(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        net.eval()
        out = net(x)
        loss = loss_func(out, y)
        print(loss.detach())
        print('----')
        print(f'--------------------------{epoch}--------------------------------------')
        time.sleep(1)
        net.train()

    if epoch % 100 == 0:
        # adjust the learning-rate
        # weight decay every 1000 epochs
        lr = optimizer.param_groups[0]['lr']
        lr *= 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

net.eval()
print(net(x).data)
print(time.time()-start_time)
print(torch.cuda.max_memory_allocated() / 1024 / 1024)
print(torch.cuda.max_memory_cached() / 1024 / 1024)


# 保存数据
torch.cuda.memory._dump_snapshot('cuda_memory3.pickle')

# 停掉记录，关闭snapshot
torch.cuda.memory._record_memory_history(enabled=None)


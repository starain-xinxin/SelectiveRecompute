import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Union
import Checkmodel as Cm
import Pretrain
import copy
import utils


def Pre_launch(model: nn.Module, dataloader: DataLoader,
               loss_fn, tolerance = 0.9,
               optim_type:Union[str, None] = None,
               CheckPointFunc: Union[str, None] = None,
               device: Union[str, None] = None,
               folder: Union[str, None] = None):
    """ 用于构造合适的 gradient—checkpoint 断点
    1. 在 cpu 上完成一个 iter 的前向
    2. 构建 model-forest，设置每个节点，计算每个节点的 profit 与 cost
    3. 计算最小化的 memory-cost 方案
    4. 在 cuda 上完成一个 iter 的前向，检测 memory
    5. 在 GPU-memory 限制下，最大化 compute-profit
    6. 返回打好断点的 model
     """
    if folder is None:
        folder = '/home/yuanxinyu/SelectiveRecompute/data/test'
    ## 1
    if optim_type is None or optim_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    else :
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        out = model(inputs)
        optimizer.zero_grad()
        loss = loss_fn(out, targets)
        loss.backward()
        optimizer.step()
        break

    ## 2
    model_forest = Pretrain.CheckForest(model)
    model_forest.make_profit_cost()

    ## 3
    min_memory_check_list = min_memory(model_forest)
    print(min_memory_check_list)

    ## 4
    # 初始化一个 memory monitor
    min_memory_monitor = utils.MemoryMonitor(folder=folder, name='min-memory', device=device, is_snapshot=True)
    min_memory_monitor.start()
    # cuda 前向一个 iter
    if optim_type is None or optim_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    else :
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.to(device)
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        out = model(inputs)
        loss = loss_fn(out, targets)
        loss.backward()
        optimizer.step()
        break
    # 结束记录
    min_memory_monitor.end()

    ## 5


def min_memory(forest: Pretrain.CheckForest):
    """ 寻找花销最小的 check point 断点 """
    check_list = []
    max_forward_buffer = 0
    # 初始化一个列表, 更新 max_forward_buffer：前向中最大的临时块buffer
    for tree in forest.check_forest:
        for node in tree.SonNodeList:
            check_list.append(node)
            if node.node_cost > max_forward_buffer:
                max_forward_buffer = node.node_cost
    # 循环
    while True:
        temp_memory = 0
        prim_memory = max_forward_buffer
        break_flag = False

        temp_list = copy.deepcopy(check_list)   # 临时拷贝一个check断点
        for max_memory_part_node in temp_list:
            # 遍历整个list
            if max_memory_part_node.node_cost == max_forward_buffer:    # 如果节点是阻碍 memory 减小的 node
                if len(max_memory_part_node.SonNodeList) == 0:               # 如果节点的子节点列表为空，需要直接break更新循环
                    break_flag = True
                    break
                prim_memory -= max_memory_part_node.node_cost           # 更新原方案的memory总数
                pop_index = temp_list.index(max_memory_part_node)
                temp_list.pop(pop_index)                                # pop 出 list
                for i in max_memory_part_node.SonNodeList:              # append 进所有子节点
                    temp_list.append(i)
                    temp_memory -= i.node_cost                          # 更新临时方案的memory总数

        if break_flag:          # 如果节点的子节点列表为空，需要直接break更新循环
            break

        temp_forward_buffer = 0
        for j in temp_list:
            if j.node_cost > temp_forward_buffer:
                temp_forward_buffer = j.node_cost
        temp_memory += temp_forward_buffer                              # 这是新方案的memory总数

        if temp_memory < prim_memory:   # 如果新方案 memory 更小
            check_list = temp_list
            max_forward_buffer = temp_forward_buffer
        else:
            break

    # 更新 model 的 self.gradient_checkpoint
    # TODO: 为了安全，这里补一个将所有 model.gradient_checkpoint 置为 False 的函数
    for node in check_list:
        node.model.gradient_checkpoint = True

    return check_list



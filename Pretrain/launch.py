import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Type, Union, Tuple
import Checkmodel as Cm
import Pretrain
import copy
import utils
import logging
import math


def Pre_launch(model: nn.Module, dataloader: DataLoader,
               loss_fn, tolerance = 0.95, max_GPU_memory = 80,
               optim_type:Union[str, None] = None,
               CheckPointFunc: Union[str, None] = None,
               device: Union[str, None] = None,
               folder: Union[str, None] = None,
               is_snapshot = False):
    """ 用于构造合适的 gradient—checkpoint 断点
    1. 在 cpu 上完成一个 iter 的前向
    2. 构建 model-forest，设置每个节点，计算每个节点的 profit 与 cost
    3. 计算最小化的 memory-cost 方案
    4. 在 cuda 上完成一个 iter 的前向，检测 memory
    5. 计算供给 check forest 使用的最大memory数值
    6. 在 GPU-memory 限制下，最大化 compute-profit
    7. 返回打好断点的 model
    :param tolerance 用于安全的 memory 衰减系数
    :param max_GPU_memory 默认A100，单位：GB
     """
    logging.info(f'---------------- Pre_launch start -----------------------')

    if folder is None:
        folder = '/home/yuanxinyu/SelectiveRecompute/data/test'
    ## ---- 1 ---------------------------------------------
    logging.info(f'cpu-iter')
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

    ## ----- 2 ---------------------------------------------
    model_forest = Pretrain.CheckForest(model)
    model_forest.make_profit_cost()

    ## ---- 3 ----------------------------------------------
    logging.info(f'min-memory-select')
    min_memory_check_list = min_memory(model_forest)
    print('使用checkpoint的模型：')
    for node in min_memory_check_list:
        print(type(node.model))

    ## ---- 4 ----------------------------------------------
    # 初始化一个 memory monitor
    logging.info(f'min-GPU-iter')
    min_memory_monitor = utils.MemoryMonitor(folder=folder, name='min-memory', device=device, is_snapshot=is_snapshot)
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
    max_tensor_memory = min_memory_monitor.max_tensor_memory
    logging.info(f'最小化 memory cost 策略下，最大张量占用：{max_tensor_memory / 1024}GB')

    ## ---- 5 -----------------------------------------------
    min_theoretical_cost = forest_memory_cost(model_forest)
    logging.info(f'最小化memory cost策略下，理论checkpoint占用缓存大小：{min_theoretical_cost / (1024**3)}GB')
    free_memory = max_GPU_memory * 1024**3 - max_tensor_memory * 1024**2 + min_theoretical_cost
    free_memory = math.floor(free_memory * tolerance)
    logging.info(f'可放置的memory大小：{free_memory / (1024**3)}GB')

    ## ---- 6 -----------------------------------------------
    free_memory_GB = free_memory/(1024**3)
    false_set = fill_up_memory(forest=model_forest, free_memory=free_memory_GB)
    normalize_forest(model_forest)
    print(f'不进行重计算的层:')
    for i in false_set:
        print(type(i.model))

    logging.info(f'---------------- Pre_launch end -----------------------')


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
    # 为了安全，这里补一个将所有 model.gradient_checkpoint 置为 False 的函数
    make_forest_type_check(forest, gradient_checkpoint=False, model_type='All')
    for node in check_list:
        node.model.gradient_checkpoint = True
        node.gradient_checkpoint = True

    return check_list

def make_forest_type_check(forest: Pretrain.CheckForest, gradient_checkpoint: bool,
                           model_type:Union[Type, Tuple[Type, ...], str] = 'All'):
    """ 根据类型(model_type)完成森林的 gradient_checkpoint 的设置 """
    tree_list = forest.check_forest
    for tree in tree_list:

        def make_node_type_check(node:Union[Pretrain.CheckModelTree, Pretrain.CheckModelNode],
                                 _gradient_checkpoint: bool, _model_type:Union[Type, Tuple[Type, ...], str] = 'All'):
            if isinstance(node, Pretrain.CheckModelTree):
                son_list = node.SonNodeList
                for SonNode in son_list:
                    make_node_type_check(SonNode, _gradient_checkpoint, _model_type)
            elif isinstance(node, Pretrain.CheckModelNode):
                if _model_type == 'All':
                    node.gradient_checkpoint = _gradient_checkpoint
                    node.model.gradient_checkpoint = _gradient_checkpoint
                    son_list = node.SonNodeList
                    for SonNode in son_list:
                        make_node_type_check(SonNode, _gradient_checkpoint, _model_type)
                else:
                    if isinstance(node.model, _model_type):
                        node.gradient_checkpoint = _gradient_checkpoint
                        node.model.gradient_checkpoint = _gradient_checkpoint
                    else:
                        node.gradient_checkpoint = not _gradient_checkpoint
                        node.model.gradient_checkpoint = _gradient_checkpoint
                    son_list = node.SonNodeList
                    for SonNode in son_list:
                        make_node_type_check(SonNode, _gradient_checkpoint, _model_type)
            else:
                assert 1, "node type error"

        make_node_type_check(tree, gradient_checkpoint, model_type)
    return forest


def forest_memory_cost(forest: Pretrain.CheckForest):
    """ 用于计算一种 checkpoint 策略下，Check forest 的理论占用显存 """
    tree_list = forest.check_forest
    max_forward_buffer = 0
    theoretical_cost = 0

    def _node_cost_recursion(node:Pretrain.CheckModelNode, level: int):
        """ 节点的递归函数 """
        nonlocal theoretical_cost, max_forward_buffer
        if not node.model.gradient_checkpoint:          # node 不做重计算，带来memory cost
            if level == 1:
                theoretical_cost += node.node_cost
            son_nodes = node.SonNodeList
            for son in son_nodes:                       # 递归
                _node_cost_recursion(son, level + 1)
        else:                                           # 如果重计算
            if node.node_cost > max_forward_buffer:     # 如果本节点峰值显存很高，那么更新峰值显存
                max_forward_buffer = node.node_cost
            if level > 1:
                theoretical_cost -= node.node_cost

    for tree in tree_list:          # 对每棵都计算cost
        CheckNode_list = tree.SonNodeList
        for node in CheckNode_list:
            _node_cost_recursion(node, 1)
    theoretical_cost += max_forward_buffer

    return theoretical_cost

def fill_up_memory(forest: Pretrain.CheckForest, free_memory, methed: str = 'greedy'):
    """ 给定 forest 和最大可接受显存 free memory , 计算放置的方法 """
    candidate_node_list = []        # 候选的断点，对应背包问题的物品列表
    candidate_profit_list = []      # 计算收益，对应背包问题中的物品价值
    candidate_cost_list = []        # 显存代价，对应背包问题中的物品价格
    temp_free_memory = free_memory  # 用于求解单个背包问题的求解
    temp_free_memory = temp_free_memory * (1024 ** 3)
    max_forward_buffer = 0
    false_set = set([])                  # 用于存放所有是False的节点

    # 首层节点(出去root节点),构造第一层的背包问题
    tree_list = forest.check_forest
    for tree_root in tree_list:
        sons_list = tree_root.SonNodeList
        for son_node in sons_list:
            candidate_node_list.append(son_node)
            candidate_cost_list.append(son_node.node_cost)
            candidate_profit_list.append(son_node.node_profit)
            if son_node.node_cost > temp_free_memory:
                max_forward_buffer = son_node.node_cost

    # 思路：构造循环，求背包问题，然后输出处理完的节点到set()中，然后构造下一层的背包问题，出口是物品列表为空的时候
    # 思路：本节点若放入背包，父亲节点一定放到背包中
    # 思路：然后，将forest全部置为True，对于在set中的节点，置False，返回node
    while len(candidate_cost_list) > 0:
        if methed == 'greedy':
            _, select_index_list = knapsack_greedy_01(candidate_cost_list, candidate_profit_list,
                                                      temp_free_memory - max_forward_buffer)      # 解背包问题
        elif methed =='back track':
            _, select_index_list = knapsack_backtrack(candidate_cost_list, candidate_profit_list,
                                                temp_free_memory - max_forward_buffer)      # 解背包问题

        select_node_list = []           # 选中的node
        for index in select_index_list:
            select_node_list.append(candidate_node_list[index])

        for node in select_node_list:   # 更新最大可放置的memory的大小
            temp_free_memory -= node.node_cost

        for node in select_node_list:   # 没选中的node
            candidate_node_list.remove(node)
        no_select_node_list = candidate_node_list

        candidate_node_list = []        # 构造新的背包问题
        candidate_cost_list = []
        candidate_profit_list = []
        max_forward_buffer = 0
        for node in no_select_node_list:
            for son_node in node.SonNodeList:
                candidate_node_list.append(son_node)
                candidate_cost_list.append(son_node.node_cost)
                candidate_profit_list.append(son_node.node_profit)
                if son_node.node_cost > temp_free_memory:
                    max_forward_buffer = son_node.node_cost

        # 选中的背包放入 set 中
        for node in select_node_list:
            false_set.add(node)

    # 全部置为True
    forest = make_forest_type_check(forest, True, 'All')

    # 递归处理 false_set
    temp_set = set([])
    for node in false_set:
        def _father_set_recursion(node: Pretrain.CheckModelNode):
            nonlocal temp_set
            father_node = node.FatherNode
            if isinstance(father_node.model, Cm.CheckModule):
                temp_set.add(father_node)
                _father_set_recursion(father_node)
        make_node_type_check1(node, False, 'All')
        _father_set_recursion(node)
    false_set = false_set | temp_set

    # 应用于forest
    for node in false_set:
        node.gradient_checkpoint = False
        node.model.gradient_checkpoint = False
    return false_set


def knapsack_backtrack(weights, values, W):
    """ 回溯法求解背包问题 """
    n = len(weights)  # 物品的数量
    max_value = 0  # 最大价值的初始值
    best_combination = []  # 最优解的初始组合

    # 回溯函数
    def backtrack(i, current_weight, current_value, current_combination):
        nonlocal max_value, best_combination  # 引用外部变量

        # 如果已经考虑了所有物品
        if i == n:
            if current_value > max_value:
                max_value = current_value  # 更新最大价值
                best_combination = current_combination[:]  # 更新最优组合
            return

        # 选择当前物品
        if current_weight + weights[i] <= W:  # 如果选择当前物品后不会超重
            current_combination.append(i)  # 将当前物品加入组合
            backtrack(i + 1, current_weight + weights[i], current_value + values[i], current_combination)  # 递归处理下一个物品
            current_combination.pop()  # 回溯，撤销选择当前物品

        # 不选择当前物品
        backtrack(i + 1, current_weight, current_value, current_combination)  # 递归处理下一个物品

    # 从第0个物品开始回溯
    backtrack(0, 0, 0, [])  # 初始状态：第0个物品，当前重量为0，当前价值为0，当前组合为空

    return max_value, best_combination  # 返回最大价值和最优组合


def knapsack_greedy_01(weights, values, W):
    """ 贪心法求解0/1背包问题（近似解法） """
    n = len(weights)  # 物品的数量
    # 计算每个物品的单位重量价值
    value_per_weight = [(values[i] / weights[i], weights[i], values[i], i) for i in range(n)]
    # 按单位重量价值从高到低排序
    value_per_weight.sort(reverse=True, key=lambda x: x[0])

    max_value = 0  # 最大价值
    current_weight = 0  # 当前重量
    best_combination = []  # 最优组合（存储物品索引）

    for vw, weight, value, idx in value_per_weight:
        if current_weight + weight <= W:
            # 如果当前物品可以完全装入背包
            current_weight += weight
            max_value += value
            best_combination.append(idx)  # 完全选中当前物品
        # 0/1背包问题不能选部分物品，所以不考虑 else 分支

    return max_value, best_combination  # 返回最大价值和最优组合

def make_node_type_check1(node:Union[Pretrain.CheckModelTree, Pretrain.CheckModelNode],
                         _gradient_checkpoint: bool, _model_type:Union[Type, Tuple[Type, ...], str] = 'All'):
    """ 本节点及其子树的type型操作 """
    if isinstance(node, Pretrain.CheckModelTree):
        son_list = node.SonNodeList
        for SonNode in son_list:
            make_node_type_check1(SonNode, _gradient_checkpoint, _model_type)
    elif isinstance(node, Pretrain.CheckModelNode):
        if _model_type == 'All':
            node.gradient_checkpoint = _gradient_checkpoint
            node.model.gradient_checkpoint = _gradient_checkpoint
            son_list = node.SonNodeList
            for SonNode in son_list:
                make_node_type_check1(SonNode, _gradient_checkpoint, _model_type)
        else:
            if isinstance(node.model, _model_type):
                node.gradient_checkpoint = _gradient_checkpoint
                node.model.gradient_checkpoint = _gradient_checkpoint
            else:
                node.gradient_checkpoint = not _gradient_checkpoint
                node.model.gradient_checkpoint = _gradient_checkpoint
            son_list = node.SonNodeList
            for SonNode in son_list:
                make_node_type_check1(SonNode, _gradient_checkpoint, _model_type)
    else:
        assert 1, "node type error"

def normalize_forest(forest: Pretrain.CheckForest):
    """ 规范化函数,对节点子树全部置为False """
    tree_list = forest.check_forest
    for tree_root in tree_list:
        SonNodes = tree_root.SonNodeList
        for son_node in SonNodes:
            def _is_node_true(node: Pretrain.CheckModelNode):
                if isinstance(node, Pretrain.CheckModelNode) and node.model.gradient_checkpoint:
                    make_node_type_check1(node, False,'All')
                    node.gradient_checkpoint = True
                    node.model.gradient_checkpoint = True
                else:
                    sons_list = node.SonNodeList
                    for son in sons_list:
                        _is_node_true(son)
            _is_node_true(son_node)





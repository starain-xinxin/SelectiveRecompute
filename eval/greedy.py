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

# 示例调用
weights = [2, 3, 5, 7]
values = [10, 5, 15, 7]
W = 10
max_value, best_combination = knapsack_greedy_01(weights, values, W)
print(f"Maximum value: {max_value}")
print(best_combination)

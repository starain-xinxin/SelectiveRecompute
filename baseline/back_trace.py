# 定义回溯算法解决背包问题
def knapsack_backtrack(weights, values, W):
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

# 示例
weights_ = [2, 2, 4, 6, 3]  # 物品的重量列表
values_ = [3, 4, 8, 9, 6]  # 物品的价值列表
W = 9  # 背包的容量

max_value_, best_combination_ = knapsack_backtrack(weights_, values_, W)  # 调用回溯算法求解
print("最大价值:", max_value_)  # 输出最大价值
print("选择的物品索引:", best_combination_)  # 输出最优组合的物品索引

import torch.nn as nn

class CheckModule(nn.Module):
    """ 参与重计算选择的模型需要构建在该类之上 """
    def __init__(self):
        super().__init__()
        self.gradient_checkpoint = False    # 是否启用重计算
        self._data_shape = None             # 输入的数据的shape
        self._data_dtype = None             # 数据的类型
        self.check_level = None             # CheckModule的递归层级

    def _compute_profit(self):
        """
        :return 计算的收益(相比于重计算需要的计算量)
        """
        raise NotImplementedError

    def _memory_cost(self):
        """
        :return:存储的代价(相比于重计算的存储)
        """
        raise NotImplementedError

    def set_level(self, level):
        """ CheckModule模型的递归层级 """
        self.check_level = level

    def _set_sub_check_level(self):
        """ 遍历子模块，设置递归深度 """
        for name, sub_model in self._modules:
            if isinstance(sub_model, CheckModule):
                sub_model.set_level(self.check_level + 1)    # 赋值下一层的level
                sub_model._set_sub_check_level()        # 让下一层继续这个操作

    def _get_sub_compute_profit(self):
        """ 返回一个list，返回下一个层级的各个 CheckModule 的 compute_profit """
        sub_compute_profit = []
        for name, sub_model in self._modules.items():
            if isinstance(sub_model, CheckModule):
                sub_compute_profit.append(sub_model._compute_profit())
        return sub_compute_profit

    def _get_sub_memory_cost(self):
        """ 返回一个list，返回下一个层级的各个 CheckModule 的 compute_profit """
        sub_mem_cost = []
        for name, sub_model in self._modules.items():
            if isinstance(sub_model, CheckModule):
                sub_mem_cost.append(sub_model._memory_cost())
        return sub_mem_cost

class TopCheckModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.check_level = 0

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def _set_sub_check_level(self):
        """ 遍历子模块，设置递归深度 """
        check_level = 0
        for name, sub_model in self._modules:
            if isinstance(sub_model, CheckModule):
                sub_model.set_level(check_level + 1)    # 赋值下一层的level
                sub_model._set_sub_check_level()        # 让下一层继续这个操作

    def set_grad_checkpoint(self):
        # TODO：这个函数是否考虑删除
        raise NotImplementedError

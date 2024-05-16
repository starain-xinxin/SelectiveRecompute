import torch.nn as nn
from typing import Union
import Checkmodel as Cm

class CheckModelNode:
    """ CheckModel Tree 模型递归树的节点 """

    def __init__(self, model:Union[None, Cm.TopCheckModule, Cm.CheckModule],
                 FatherNode:Union[None, 'CheckModelNode']):
        # 模型,命名,check_level
        if isinstance(model, Cm.TopCheckModule) or \
            isinstance(model, Cm.CheckModule) or \
            model is None:
            self.model = model
            if model is not None:
                self.short_name =f'{type(model).__name__}'
                self.check_level = model.check_level
            else:
                self.short_name = None
                self.check_level = None
        else:
            assert 0, 'model is not TopCheckModule or CheckModule'
        # 节点信息
        self.FatherNode = FatherNode
        self.SonNodeList = []
        self.is_root = False
        self.is_leaf = True

    def _Create_SonNode(self):
        for name, sub_model in self.model._modules.items():
            if isinstance(sub_model, Cm.CheckModule):   # 如果有子model
                node = CheckModelNode(sub_model, self)  # 创建新的节点
                self.is_leaf = False                    # 本节点不再是叶子节点
                self.SonNodeList.append(node)           # 加入到孩子列表中
                node._Create_SonNode()                  # 对孩子也进行递归操作
            elif isinstance(sub_model, nn.ModuleList):
                for sub_sub_model in sub_model:
                    if isinstance(sub_sub_model, Cm.CheckModule):   # 如果有子model
                        node = CheckModelNode(sub_sub_model, self)  # 创建新的节点
                        self.is_leaf = False                    # 本节点不再是叶子节点
                        self.SonNodeList.append(node)           # 加入到孩子列表中
                        node._Create_SonNode()                  # 对孩子也进行递归操作

    def _make_profit_cost(self):
        """ 递归计算节点的profit与cost """
        if isinstance(self.model, Cm.CheckModule):
            # 如果 CheckModule, 就计算本节点的 profit 和 cost
            self.node_profit = self.model._compute_profit()
            self.node_cost = self.model._memory_cost()
        for son_node in self.SonNodeList:
            son_node._make_profit_cost()    # 递归计算


class CheckModelTree(CheckModelNode):
    """ 模型递归树：实际是根节点，TopCheckModule """

    def __init__(self, model:Cm.TopCheckModule):
        if not isinstance(model, Cm.TopCheckModule):
            assert 0, 'CheckModelTree\'model must be TopCheckModule'
        super().__init__(model, FatherNode=None)
        self.is_root = True
        super()._Create_SonNode()

    def _make_profit_cost(self):
        super()._make_profit_cost()


class CheckForest:
    """ 从 nn.Module 中构建所有的 CheckModelTree """

    def __init__(self, Model:nn.Module):
        check_forest = []
        def create(model):
            nonlocal  check_forest
            if isinstance(model, Cm.TopCheckModule):
                tree = CheckModelTree(model)
                check_forest.append(tree)
            else:
                for name, sub_model in model._modules.items():
                    create(sub_model)
        create(Model)
        self.num_trees = len(check_forest)
        self.check_forest = check_forest    # 存一簇 TopCheckModule

    def make_profit_cost(self):
        """ 对模型进行一次cpu前向后，调用此方法来获取 计算profit 和 memory cost """
        for tree in self.check_forest:
            tree._make_profit_cost()





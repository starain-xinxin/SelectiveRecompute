import torch.nn as nn
import torch.utils.checkpoint as Cp
import Checkmodel.CheckModule as CheckModule

class FNN(CheckModule, nn.Module):
    """ FNN: 双层DNN，中间的激活层默认ReLU """
    def __init__(self, input_dim ,hidden_dim, out_dim=None, activation_func='ReLU', gradient_checkpoint=False):
        """
        :param input_dim: 输入层
        :param hidden_dim: 隐藏层
        :param out_dim: 输出层(默认输入层)
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.fnn1 = nn.Linear(input_dim, hidden_dim)
        self.fn = eval(f'nn.{activation_func}()')
        if out_dim is None:
            out_dim = input_dim
        self.out_dim = out_dim
        self.fnn2 = nn.Linear(hidden_dim, out_dim)
        self.gradient_checkpoint = gradient_checkpoint
        self._data_shape = None
        self._data_dtype = None

    def forward(self, x):
        if not x.requires_grad:
            x.requires_grad_()
        if self._data_shape is None:
            self._data_shape = list(x.size())
        if self._data_dtype is None:
            self._data_dtype = x.dtype
        if self.gradient_checkpoint:
            x = Cp.checkpoint(self._checkpoint_forward, x, use_reentrant=False)
        else:
            x = self._checkpoint_forward(x)
        return x

    def _checkpoint_forward(self, x):
        x = self.fnn1(x)
        x = self.fn(x)
        x = self.fnn2(x)
        return x

    def _compute_profit(self):
        """ b*s * h * (input + output) """
        data_shape = self._data_shape[0] * self._data_shape[1]
        return  data_shape * self.hidden_dim * (self.input_dim  +  self.out_dim)

    def _memory_cost(self):
        """ b*s * h * 2 """
        dtype_bytes = self._data_dtype.itemsize
        data_shape = self._data_shape[0] * self._data_shape[1]
        return data_shape * self.hidden_dim * 2 * dtype_bytes
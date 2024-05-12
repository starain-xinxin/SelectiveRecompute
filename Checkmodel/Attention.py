import torch
import torch.nn as nn
import torch.utils.checkpoint as Cp
import Checkmodel.CheckModule as CheckModule
import Checkmodel as Cm


class MultiHeadQKV(CheckModule, nn.Module):
    """ 用于QKV矩阵相乘,这里修改为一个head的类 """
    def __init__(self, d_model, dim_k=None, dim_v=None, gradient_checkpoint=False):
        super().__init__()
        self.d_model = d_model
        self.num_head = 1
        if dim_k is None:
            dim_k = d_model
        self.dim_k = dim_k
        self.dim_q = dim_k
        if dim_v is None:
            dim_v = dim_k
        self.dim_v = dim_v

        self.Wq = nn.Linear(self.d_model, self.num_heads*self.dim_q, bias=False)
        self.Wk = nn.Linear(self.d_model, self.num_heads*self.dim_k, bias=False)
        self.Wv = nn.Linear(self.d_model, self.num_heads*self.dim_v, bias=False)

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
        matrix_Q = self.Wq(x)
        matrix_K = self.Wk(x)
        matrix_V = self.Wv(x)
        Attn_Matrix = matrix_Q @ matrix_K.T
        return matrix_V, Attn_Matrix

    def _compute_profit(self):
        data_shape = self._data_shape[0] * self._data_shape[1]
        compute_Q_or_K = data_shape * self.num_head * self.dim_k
        compute_V = data_shape * self.num_head * self.dim_v
        compute_QKT = data_shape * self.dim_q ** 2
        return 2 * compute_Q_or_K + compute_V + compute_QKT

    def _memory_cost(self):
        dtype_bytes = self._data_dtype.itemsize     # 一个张量元素占据的字节数
        data_shape = self._data_shape[0] * self._data_shape[1]
        mem_K_or_Q = data_shape * self.num_head * self.dim_k
        return 2 * mem_K_or_Q * dtype_bytes

class AttentionCore(CheckModule, nn.Module):
    """ 计算 attn core """
    def __init__(self, dim_k, gradient_checkpoint=False):
        super().__init__()
        self.dim_k = dim_k
        self.softmax = nn.Softmax()
        self.gradient_checkpoint = gradient_checkpoint
        self._data_shape = None
        self._data_dtype = None

    def forward(self, matrix_V, Attn_Matrix, mask=None):
        if not matrix_V.requires_grad or not Attn_Matrix.requires_grad:
            matrix_V.requires_grad_()
            Attn_Matrix.requires_grad_()
        if self._data_shape is None:
            shape1 = list(matrix_V.size())
            shape2 = list(Attn_Matrix.size())
            self._data_shape = []
            self._data_shape.append(shape1)
            self._data_shape.append(shape2)
        if self._data_dtype is None:
            self._data_dtype = matrix_V.dtype

        if self.gradient_checkpoint:
            x = Cp.checkpoint(self._checkpoint_forward, matrix_V, Attn_Matrix, use_reentrant=False)
        else:
            x = self._checkpoint_forward(matrix_V, Attn_Matrix)
        return x

    def _checkpoint_forward(self, matrix_V, Attn_Matrix, mask=None):
        sqrt = torch.sqrt(torch.tensor([self.dim_k], requires_grad=True))
        if mask is None:
            mask = 0       # TODO:完善mask的默认值
        core = Attn_Matrix / sqrt + mask
        core = self.softmax(core)
        return core @ matrix_V

    def _compute_profit(self):
        batch_size = self._data_shape[0][0]
        seq_len = self._data_shape[0][1]
        dim_v = self._data_shape[1][-1]
        return batch_size * seq_len * dim_v

    def _memory_cost(self):
        dtype_bytes = self._data_dtype.itemsize     # 一个张量元素占据的字节数
        batch_size = self._data_shape[0][0]
        seq_len = self._data_shape[0][1]
        dim_v = self._data_shape[1][-1]
        softmax_mem = batch_size * seq_len * seq_len
        mask_mem = softmax_mem
        sqrt_mem = softmax_mem
        return softmax_mem + mask_mem + sqrt_mem

class MultiAttention(Cm.CheckModule, nn.Module):
    """ 多头注意力机制 """
    def __init__(self,d_model, dim_k, dim_v, num_head=1, gradient_checkpoint=False):
        super().__init__()
        self.gradient_checkpoint = gradient_checkpoint
        self._data_shape = None
        self._data_dtype = None
        # TODO: 这里可以增加默认选择
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_head = num_head
        self.d_model = d_model
        self.QKVList = nn.ModuleList([MultiHeadQKV(self.d_model, dim_k=self.dim_k, dim_v=self.dim_v) for i in range(self.num_head)])
        self.CoreList = nn.ModuleList([AttentionCore(self.dim_k) for i in range(self.num_head)])
        self.CatMatrix = nn.Linear(self.num_head * self.dim_k, self.d_model)


    def _checkpoint_forward(self, x, mask=None):
        Attn_core_list = []
        # 计算多个注意力头的结果，存入 Attn_core_list
        for QKV_Key, Core_key in zip(self.QKVList.keys(), self.CoreList.keys()):
            # matrix_V [batch, seq_len, dim_v]   Attn_Matrix [batch, seq_len, seq_len]
            matrix_V, Attn_Matrix = self.QKVList[QKV_Key](x)
            attn_core = self.CoreList[Core_key](matrix_V, Attn_Matrix, mask)
            Attn_core_list.append(attn_core)
        # concat 结果
        Attn_Core = torch.cat(Attn_core_list, dim=-1)   # tensor (batch, seq_len, heads*dim_v)
        # 转换
        Attn_Core = self.CatMatrix(Attn_Core)
        return Attn_Core

    def forward(self, x, mask=None):
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

    def _compute_profit(self):
        batch = self._data_shape[0]
        seq_len = self._data_shape[1]
        # 子模块
        sub_compute_profit_list = self._get_sub_compute_profit()
        sub_compute_profit = 0
        for i in sub_compute_profit_list:
            sub_compute_profit += i
        # 本模块
        self_compute_profit = batch * seq_len * (self.num_head * self.dim_v)**2 * self.d_model
        return sub_compute_profit + self_compute_profit

    def _memory_cost(self):
        dtype_bytes = self._data_dtype.itemsize     # 一个张量元素占据的字节数
        batch = self._data_shape[0]
        seq_len = self._data_shape[1]
        # 子模块
        sub_mem_cost_list = self._get_sub_memory_cost()
        sub_mem_cost = 0
        for i in sub_mem_cost_list:
            sub_mem_cost += i
        # 本模块
        core_out_cost_matrix_V = self.num_head * batch * seq_len * self.dim_v * dtype_bytes
        core_out_cost_Attn_Matrix = self.num_head * batch * seq_len * seq_len * dtype_bytes
        return sub_mem_cost + 2 * (core_out_cost_Attn_Matrix + core_out_cost_matrix_V)

class CheckEncoder(Cm.TopCheckModule, nn.Module):
    """ Encoder 编码器 """
    def __init__(self, d_model, dim_k, dim_v, num_head=1, hidden_dim=None, activation_func='ReLU'):
        super().__init__()
        self.d_model = d_model
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_head = num_head
        if hidden_dim is None:
            hidden_dim = d_model
        self.hidden_dim = hidden_dim
        self.activation_func = activation_func

        self.MultiAttention = MultiAttention(self.d_model, self.dim_k, self.dim_v, self.num_head)
        self.LN1 = Cm.LayerNormal(self.d_model)

        self.FFN = Cm.FNN(self.d_model, hidden_dim=self.hidden_dim, activation_func=self.activation_func)
        self.LN2 = Cm.LayerNormal(self.d_model)

    def forward(self, x):
        # attention 部分
        x1 = self.LN1(x)
        x1 = self.MultiAttention(x1)
        x = x + x1
        # fnn部分
        x2 = self.LN2(x)
        x2 = self.FFN(x2)
        x = x + x2
        return x





















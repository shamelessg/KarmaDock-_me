#!usr/bin/env python3
# -*- coding:utf-8 -*-
'''
@文件    :   GVP Block
@时间    :   2022/10/13 10:35:49
@作者    :   Xujun Zhang
@版本    :   1.0
@联系方式 :   
@许可证   :   
@描述    :   无
'''

import functools
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add


class GVP_embedding(nn.Module):
    '''
    基于https://github.com/drorlab/gvp-pytorch/blob/main/gvp/models.py修改
    如论文中所述的用于模型质量评估的GVP-GNN。

    接收`torch_geometric.data.Data`或`torch_geometric.data.Batch`类型的蛋白质结构图，
    并返回批次中每个图的标量分数，形状为[n_nodes]的`torch.Tensor`

    应与`gvp.data.ProteinGraphDataset`一起使用，或与具有相同属性的`torch_geometric.data.Batch`对象生成器一起使用。

    :param node_in_dim: 输入图中的节点维度，如果使用原始特征应为(6, 3)
    :param node_h_dim: GVP-GNN层中使用的节点维度
    :param node_in_dim: 输入图中的边维度，如果使用原始特征应为(32, 1)
    :param edge_h_dim: 在GVP-GNN层中使用前要嵌入的边维度
    :seq_in: 如果为`True`，序列也将在前向传递中传入；否则，序列信息被假定为输入节点嵌入的一部分
    :param num_layers: GVP-GNN层的数量
    :param drop_rate: 所有 dropout 层中使用的速率
    '''

    def __init__(self, node_in_dim, node_h_dim,
                 edge_in_dim, edge_h_dim,
                 seq_in=False, num_layers=3, drop_rate=0.1):

        super(GVP_embedding, self).__init__()

        if seq_in:
            self.W_s = nn.Embedding(31, 31)
            node_in_dim = (node_in_dim[0] + 31, node_in_dim[1])

        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None))
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None))
        )

        self.layers = nn.ModuleList(
            GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate)
            for _ in range(num_layers))

        ns, _ = node_h_dim
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim),
            GVP(node_h_dim, (ns, 0)))

    def forward(self, h_V, edge_index, h_E, seq):
        '''
        :param h_V: 节点嵌入的元组 (s, V)
        :param edge_index: 形状为 [2, num_edges] 的 `torch.Tensor`
        :param h_E: 边嵌入的元组 (s, V)
        :param seq: 如果不为 `None`，则为形状为 [num_nodes] 的 int `torch.Tensor`
                    将被嵌入并附加到 `h_V`
        '''
        seq = self.W_s(seq)
        h_V = (torch.cat([h_V[0], seq], dim=-1), h_V[1])
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)
        out = self.W_out(h_V)

        return out


def tuple_sum(*args):
    '''
    按元素对任意数量的元组 (s, V) 求和。
    '''
    return tuple(map(sum, zip(*args)))


def tuple_cat(*args, dim=-1):
    '''
    按元素连接任意数量的元组 (s, V)。

    :param dim: 当被视为标量通道张量的 `dim` 索引时，要沿其连接的维度。
                这意味着 `dim=-1` 将被应用为向量通道张量的 `dim=-2`。
    '''
    dim %= len(args[0][0].shape)
    s_args, v_args = list(zip(*args))
    return torch.cat(s_args, dim=dim), torch.cat(v_args, dim=dim)


def tuple_index(x, idx):
    '''
    沿第一维度索引到元组 (s, V) 中。

    :param idx: 任何可以用于索引 `torch.Tensor` 的对象
    '''
    return x[0][idx], x[1][idx]


def randn(n, dims, device="cpu"):
    '''
    返回从正态分布中按元素绘制的随机元组 (s, V)。

    :param n: 数据点数量
    :param dims: 维度元组 (n_scalar, n_vector)

    :return: (s, V)，其中 s.shape = (n, n_scalar) 且
             V.shape = (n, n_vector, 3)
    '''
    return torch.randn(n, dims[0], device=device), \
           torch.randn(n, dims[1], 3, device=device)


def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    '''
    张量的L2范数，钳制在最小值 `eps` 以上。

    :param sqrt: 如果为 `False`，返回L2范数的平方
    '''
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out


def _split(x, nv):
    '''
    将合并的 (s, V) 表示拆分为元组。
    应仅与 `_merge(s, V)` 一起使用，且仅在无法使用元组表示时使用。

    :param x: 从 `_merge` 返回的 `torch.Tensor`
    :param nv: `_merge` 输入中的向量通道数
    '''
    v = torch.reshape(x[..., -3 * nv:], x.shape[:-1] + (nv, 3))
    s = x[..., :-3 * nv]
    return s, v


def _merge(s, v):
    '''
    将元组 (s, V) 合并为单个 `torch.Tensor`，其中向量通道被展平并附加到标量通道。
    应仅在无法使用元组表示时使用。
    使用 `_split(x, nv)` 来反转。
    '''
    v = torch.reshape(v, v.shape[:-2] + (3 * v.shape[-2],))
    return torch.cat([s, v], -1)


class GVP(nn.Module):
    '''
    几何向量感知器。有关更多详细信息，请参阅论文和README.md。

    :param in_dims: 元组 (n_scalar, n_vector)
    :param out_dims: 元组 (n_scalar, n_vector)
    :param h_dim: 中间向量通道数，可选
    :param activations: 函数元组 (scalar_act, vector_act)
    :param vector_gate: 是否使用向量门控。
                        如果为 `True`，vector_act 将用作向量门控中的 sigma^+
    '''

    def __init__(self, in_dims, out_dims, h_dim=None,
                 activations=(F.relu, torch.sigmoid), vector_gate=False):
        super(GVP, self).__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.vector_gate = vector_gate
        if self.vi:
            self.h_dim = h_dim or max(self.vi, self.vo)
            self.wh = nn.Linear(self.vi, self.h_dim, bias=False)
            self.ws = nn.Linear(self.h_dim + self.si, self.so)
            if self.vo:
                self.wv = nn.Linear(self.h_dim, self.vo, bias=False)
                if self.vector_gate: self.wsv = nn.Linear(self.so, self.vo)
        else:
            self.ws = nn.Linear(self.si, self.so)

        self.scalar_act, self.vector_act = activations
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        '''
        :param x: `torch.Tensor` 的元组 (s, V)，
                  或（如果 vectors_in 为 0），单个 `torch.Tensor`
        :return: `torch.Tensor` 的元组 (s, V)，
                 或（如果 vectors_out 为 0），单个 `torch.Tensor`
        '''
        if self.vi:
            s, v = x
            v = torch.transpose(v, -1, -2)
            vh = self.wh(v)
            vn = _norm_no_nan(vh, axis=-2)
            s = self.ws(torch.cat([s, vn], -1))
            if self.vo:
                v = self.wv(vh)
                v = torch.transpose(v, -1, -2)
                if self.vector_gate:
                    if self.vector_act:
                        gate = self.wsv(self.vector_act(s))
                    else:
                        gate = self.wsv(s)
                    v = v * torch.sigmoid(gate).unsqueeze(-1)
                elif self.vector_act:
                    v = v * self.vector_act(
                        _norm_no_nan(v, axis=-1, keepdims=True))
        else:
            s = self.ws(x)
            if self.vo:
                v = torch.zeros(s.shape[0], self.vo, 3,
                                device=self.dummy_param.device)
        if self.scalar_act:
            s = self.scalar_act(s)

        return (s, v) if self.vo else s


class _VDropout(nn.Module):
    '''
    向量通道 dropout，其中每个向量通道的元素一起被丢弃。
    '''

    def __init__(self, drop_rate):
        super(_VDropout, self).__init__()
        self.drop_rate = drop_rate
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        '''
        :param x: 对应于向量通道的 `torch.Tensor`
        '''
        device = self.dummy_param.device
        if not self.training:
            return x
        mask = torch.bernoulli(
            (1 - self.drop_rate) * torch.ones(x.shape[:-1], device=device)
        ).unsqueeze(-1)
        x = mask * x / (1 - self.drop_rate)
        return x


class Dropout(nn.Module):
    '''
    用于元组 (s, V) 的组合 dropout。
    接收元组 (s, V) 作为输入和输出。
    '''

    def __init__(self, drop_rate):
        super(Dropout, self).__init__()
        self.sdropout = nn.Dropout(drop_rate)
        self.vdropout = _VDropout(drop_rate)

    def forward(self, x):
        '''
        :param x: `torch.Tensor` 的元组 (s, V)，
                  或单个 `torch.Tensor`
                  （将被假定为标量通道）
        '''
        if type(x) is torch.Tensor:
            return self.sdropout(x)
        s, v = x
        return self.sdropout(s), self.vdropout(v)


class LayerNorm(nn.Module):
    '''
    用于元组 (s, V) 的组合 LayerNorm。
    接收元组 (s, V) 作为输入和输出。
    '''

    def __init__(self, dims):
        super(LayerNorm, self).__init__()
        self.s, self.v = dims
        self.scalar_norm = nn.LayerNorm(self.s)

    def forward(self, x):
        '''
        :param x: `torch.Tensor` 的元组 (s, V)，
                  或单个 `torch.Tensor`
                  （将被假定为标量通道）
        '''
        if not self.v:
            return self.scalar_norm(x)
        s, v = x
        vn = _norm_no_nan(v, axis=-1, keepdims=True, sqrt=False)
        vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True))
        return self.scalar_norm(s), v / vn


class GVPConv(MessagePassing):
    '''
    使用几何向量感知器的图卷积/消息传递。
    接收带有节点和边嵌入的图，
    并返回新的节点嵌入。

    这不执行残差更新和逐点前馈层
    ---参见 `GVPConvLayer`。

    :param in_dims: 输入节点嵌入维度 (n_scalar, n_vector)
    :param out_dims: 输出节点嵌入维度 (n_scalar, n_vector)
    :param edge_dims: 输入边嵌入维度 (n_scalar, n_vector)
    :param n_layers: 消息函数中GVPs的数量
    :param module_list: 预构建的消息函数，覆盖n_layers
    :param aggr: 如果一些入边被掩码，如在掩码自回归解码器架构中，应为"add"，否则为"mean"
    :param activations: 在GVPs中使用的函数元组 (scalar_act, vector_act)
    :param vector_gate: 是否使用向量门控。
                        如果为 `True`，vector_act 将用作向量门控中的 sigma^+
    '''

    def __init__(self, in_dims, out_dims, edge_dims,
                 n_layers=3, module_list=None, aggr="mean",
                 activations=(F.relu, torch.sigmoid), vector_gate=False):
        super(GVPConv, self).__init__(aggr=aggr)
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.se, self.ve = edge_dims

        GVP_ = functools.partial(GVP,
                                 activations=activations, vector_gate=vector_gate)

        module_list = module_list or []
        if not module_list:
            if n_layers == 1:
                module_list.append(
                    GVP_((2 * self.si + self.se, 2 * self.vi + self.ve),
                         (self.so, self.vo), activations=(None, None)))
            else:
                module_list.append(
                    GVP_((2 * self.si + self.se, 2 * self.vi + self.ve), out_dims)
                )
                for i in range(n_layers - 2):
                    module_list.append(GVP_(out_dims, out_dims))
                module_list.append(GVP_(out_dims, out_dims,
                                        activations=(None, None)))
        self.message_func = nn.Sequential(*module_list)

    def forward(self, x, edge_index, edge_attr):
        '''
        :param x: `torch.Tensor` 的元组 (s, V)
        :param edge_index: 形状为 [2, n_edges] 的数组
        :param edge_attr: `torch.Tensor` 的元组 (s, V)
        '''
        x_s, x_v = x
        message = self.propagate(edge_index,
                                 s=x_s, v=x_v.reshape(x_v.shape[0], 3 * x_v.shape[1]),
                                 edge_attr=edge_attr)
        return _split(message, self.vo)

    def message(self, s_i, v_i, s_j, v_j, edge_attr):
        v_j = v_j.view(v_j.shape[0], v_j.shape[1] // 3, 3)
        v_i = v_i.view(v_i.shape[0], v_i.shape[1] // 3, 3)
        message = tuple_cat((s_j, v_j), edge_attr, (s_i, v_i))
        message = self.message_func(message)
        return _merge(*message)


class GVPConvLayer(nn.Module):
    '''
    带有几何向量感知器的完整图卷积/消息传递层。
    使用聚合的传入消息残差更新节点嵌入，
    对节点嵌入应用逐点前馈网络，并返回更新后的节点嵌入。

    要仅计算聚合消息，请参见 `GVPConv`。

    :param node_dims: 节点嵌入维度 (n_scalar, n_vector)
    :param edge_dims: 输入边嵌入维度 (n_scalar, n_vector)
    :param n_message: 消息函数中使用的GVPs数量
    :param n_feedforward: 前馈函数中使用的GVPs数量
    :param drop_rate: 所有dropout层中的丢弃概率
    :param autoregressive: 如果为 `True`，此 `GVPConvLayer` 将用于
           消息的不同输入节点嵌入集，其中 src >= dst
    :param activations: 在GVPs中使用的函数元组 (scalar_act, vector_act)
    :param vector_gate: 是否使用向量门控。
                        如果为 `True`，vector_act 将用作向量门控中的 sigma^+
    '''

    def __init__(self, node_dims, edge_dims,
                 n_message=3, n_feedforward=2, drop_rate=.1,
                 autoregressive=False,
                 activations=(F.relu, torch.sigmoid), vector_gate=False):

        super(GVPConvLayer, self).__init__()
        self.conv = GVPConv(node_dims, node_dims, edge_dims, n_message,
                            aggr="add" if autoregressive else "mean",
                            activations=activations, vector_gate=vector_gate)
        GVP_ = functools.partial(GVP,
                                 activations=activations, vector_gate=vector_gate)
        self.norm = nn.ModuleList([LayerNorm(node_dims) for _ in range(2)])
        self.dropout = nn.ModuleList([Dropout(drop_rate) for _ in range(2)])

        ff_func = []
        if n_feedforward == 1:
            ff_func.append(GVP_(node_dims, node_dims, activations=(None, None)))
        else:
            hid_dims = 4 * node_dims[0], 2 * node_dims[1]
            ff_func.append(GVP_(node_dims, hid_dims))
            for i in range(n_feedforward - 2):
                ff_func.append(GVP_(hid_dims, hid_dims))
            ff_func.append(GVP_(hid_dims, node_dims, activations=(None, None)))
        self.ff_func = nn.Sequential(*ff_func)

    def forward(self, x, edge_index, edge_attr,
                autoregressive_x=None, node_mask=None):
        '''
        :param x: `torch.Tensor` 的元组 (s, V)
        :param edge_index: 形状为 [2, n_edges] 的数组
        :param edge_attr: `torch.Tensor` 的元组 (s, V)
        :param autoregressive_x: `torch.Tensor` 的元组 (s, V)。
                如果不为 `None`，将用作形成消息的源节点嵌入，其中 src >= dst。
                当前节点嵌入 `x` 仍将是更新和逐点前馈的基础。
        :param node_mask: 类型为 `bool` 的数组，用于索引到节点嵌入 (s, V) 的第一维。
                如果不为 `None`，只有这些节点会被更新。
        '''

        if autoregressive_x is not None:
            src, dst = edge_index
            mask = src < dst
            edge_index_forward = edge_index[:, mask]
            edge_index_backward = edge_index[:, ~mask]
            edge_attr_forward = tuple_index(edge_attr, mask)
            edge_attr_backward = tuple_index(edge_attr, ~mask)

            dh = tuple_sum(
                self.conv(x, edge_index_forward, edge_attr_forward),
                self.conv(autoregressive_x, edge_index_backward, edge_attr_backward)
            )

            count = scatter_add(torch.ones_like(dst), dst,
                                dim_size=dh[0].size(0)).clamp(min=1).unsqueeze(-1)

            dh = dh[0] / count, dh[1] / count.unsqueeze(-1)

        else:
            dh = self.conv(x, edge_index, edge_attr)

        if node_mask is not None:
            x_ = x
            x, dh = tuple_index(x, node_mask), tuple_index(dh, node_mask)

        x = self.norm[0](tuple_sum(x, self.dropout[0](dh)))

        dh = self.ff_func(x)
        x = self.norm[1](tuple_sum(x, self.dropout[1](dh)))

        if node_mask is not None:
            x_[0][node_mask], x_[1][node_mask] = x[0], x[1]
            x = x_
        return x

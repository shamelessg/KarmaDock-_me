#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件    :   GraphTransformer.py
@时间    :   2022/10/30 21:02:39
@作者    :   Chao Shen
@版本    :   1.0
@联系方式 :   
@许可证   :   
@描述    :   无
'''

# 这里放入导入库
import torch as th
import torch.nn.functional as F
import copy
import numpy as np
import random
from torch_scatter import scatter_mean, scatter_add
from torch_geometric.nn import MessagePassing
from torch_sparse import matmul
import torch_geometric.transforms as T
from torch_geometric.utils import normalized_cut, to_dense_batch
from torch_geometric.nn import MetaLayer
from torch import nn
import pandas as pd

def glorot_orthogonal(tensor, scale):
	"""根据正交Glorot初始化方案初始化张量的值。"""
	if tensor is not None:
		th.nn.init.orthogonal_(tensor.data)
		scale /= ((tensor.size(-2) + tensor.size(-1)) * tensor.var())
		tensor.data *= scale.sqrt()


class MultiHeadAttentionLayer(nn.Module):
	"""使用DGLGraph的节点和边（几何）特征计算注意力分数。"""
	def __init__(self, num_input_feats, num_output_feats,
				num_heads, using_bias=False, update_edge_feats=True):
		super(MultiHeadAttentionLayer, self).__init__()
		
		# 声明共享变量
		self.num_output_feats = num_output_feats
		self.num_heads = num_heads
		self.using_bias = using_bias
		self.update_edge_feats = update_edge_feats
		
		# 定义节点特征的查询、键和值张量，以及边特征的投影张量
		self.Q = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)
		self.K = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)
		self.V = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)
		self.edge_feats_projection = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)
		
		self.reset_parameters()
		
	def reset_parameters(self):
		"""重新初始化可学习参数。"""
		scale = 2.0
		if self.using_bias:
			glorot_orthogonal(self.Q.weight, scale=scale)
			self.Q.bias.data.fill_(0)
			
			glorot_orthogonal(self.K.weight, scale=scale)
			self.K.bias.data.fill_(0)
			
			glorot_orthogonal(self.V.weight, scale=scale)
			self.V.bias.data.fill_(0)
			
			glorot_orthogonal(self.edge_feats_projection.weight, scale=scale)
			self.edge_feats_projection.bias.data.fill_(0)
		else:
			glorot_orthogonal(self.Q.weight, scale=scale)
			glorot_orthogonal(self.K.weight, scale=scale)
			glorot_orthogonal(self.V.weight, scale=scale)
			glorot_orthogonal(self.edge_feats_projection.weight, scale=scale)
	
	def propagate_attention(self, edge_index, node_feats_q, node_feats_k, node_feats_v, edge_feats_projection):
		row, col = edge_index
		e_out = None
		# 计算注意力分数
		alpha = node_feats_k[row] * node_feats_q[col]
		# 缩放和裁剪注意力分数
		alpha = (alpha / np.sqrt(self.num_output_feats)).clamp(-5.0,5.0)
		# 使用可用的边特征修改注意力分数
		alpha = alpha * edge_feats_projection
		# 复制边特征作为e_out传递给edge_feats_MLP
		if self.update_edge_feats:
			e_out = alpha
		
		# 对注意力分数应用softmax，然后裁剪	
		alphax = th.exp((alpha.sum(-1, keepdim=True)).clamp(-5.0,5.0))
		# 将加权值发送到目标节点
		wV = scatter_add(node_feats_v[row]*alphax, col, dim=0, dim_size=node_feats_q.size(0))
		z = scatter_add(alphax, col, dim=0, dim_size=node_feats_q.size(0))
		return wV, z, e_out
	
	def forward(self, x, edge_attr, edge_index):
		node_feats_q = self.Q(x).view(-1, self.num_heads, self.num_output_feats)
		node_feats_k = self.K(x).view(-1, self.num_heads, self.num_output_feats)
		node_feats_v = self.V(x).view(-1, self.num_heads, self.num_output_feats)
		edge_feats_projection = self.edge_feats_projection(edge_attr).view(-1, self.num_heads, self.num_output_feats)	
		wV, z, e_out = self.propagate_attention(edge_index, node_feats_q, node_feats_k, node_feats_v, edge_feats_projection)
		
		h_out = wV / (z + th.full_like(z, 1e-6))
		return h_out, e_out	


class GraphTransformerModule(nn.Module):
	"""图变换器模块（相当于一层图卷积）。"""
	def __init__(
										self,
										num_hidden_channels,
										activ_fn=nn.SiLU(),
										residual=True,
										num_attention_heads=4,
										norm_to_apply='batch',
										dropout_rate=0.1,
										num_layers=4,
										):
		super(GraphTransformerModule, self).__init__()
		
		# 记录给定的参数
		self.activ_fn = activ_fn
		self.residual = residual
		self.num_attention_heads = num_attention_heads
		self.norm_to_apply = norm_to_apply
		self.dropout_rate = dropout_rate
		self.num_layers = num_layers
		
		# --------------------
		# 变换器模块
		# --------------------
		# 定义与几何变换器模块相关的所有模块
		self.apply_layer_norm = 'layer' in self.norm_to_apply.lower()
		
		self.num_hidden_channels, self.num_output_feats = num_hidden_channels, num_hidden_channels
		if self.apply_layer_norm:
			self.layer_norm1_node_feats = nn.LayerNorm(self.num_output_feats)
			self.layer_norm1_edge_feats = nn.LayerNorm(self.num_output_feats)
		else:  # 否则，默认使用批归一化
			self.batch_norm1_node_feats = nn.BatchNorm1d(self.num_output_feats)
			self.batch_norm1_edge_feats = nn.BatchNorm1d(self.num_output_feats)
		
		self.mha_module = MultiHeadAttentionLayer(
			self.num_hidden_channels,
			self.num_output_feats // self.num_attention_heads,
			self.num_attention_heads,
			self.num_hidden_channels != self.num_output_feats,  # 只有当Linear()必须改变大小时才使用偏置
			update_edge_feats=True
		)
		
		self.O_node_feats = nn.Linear(self.num_output_feats, self.num_output_feats)
		self.O_edge_feats = nn.Linear(self.num_output_feats, self.num_output_feats)
		
		# 节点特征的MLP
		dropout = nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0.0 else nn.Identity()
		self.node_feats_MLP = nn.ModuleList([
			nn.Linear(self.num_output_feats, self.num_output_feats * 2, bias=False),
			self.activ_fn,
			dropout,
			nn.Linear(self.num_output_feats * 2, self.num_output_feats, bias=False)
		])
		
		if self.apply_layer_norm:
			self.layer_norm2_node_feats = nn.LayerNorm(self.num_output_feats)
			self.layer_norm2_edge_feats = nn.LayerNorm(self.num_output_feats)
		else:  # 否则，默认使用批归一化
			self.batch_norm2_node_feats = nn.BatchNorm1d(self.num_output_feats)
			self.batch_norm2_edge_feats = nn.BatchNorm1d(self.num_output_feats)
		
		# 边特征的MLP
		self.edge_feats_MLP = nn.ModuleList([
			nn.Linear(self.num_output_feats, self.num_output_feats * 2, bias=False),
			self.activ_fn,
			dropout,
			nn.Linear(self.num_output_feats * 2, self.num_output_feats, bias=False)
		])
		
		self.reset_parameters()
	
	def reset_parameters(self):
		"""重新初始化可学习参数。"""
		scale = 2.0
		glorot_orthogonal(self.O_node_feats.weight, scale=scale)
		self.O_node_feats.bias.data.fill_(0)
		glorot_orthogonal(self.O_edge_feats.weight, scale=scale)
		self.O_edge_feats.bias.data.fill_(0)
		
		for layer in self.node_feats_MLP:
			if hasattr(layer, 'weight'):  # 跳过激活函数的初始化
				glorot_orthogonal(layer.weight, scale=scale)
		
		for layer in self.edge_feats_MLP:
			if hasattr(layer, 'weight'):
				glorot_orthogonal(layer.weight, scale=scale)
	
	def run_gt_layer(self, edge_index, node_feats, edge_feats):
		"""使用多头注意力（MHA）模块执行几何注意力的前向传递。"""
		node_feats_in1 = node_feats  # 缓存节点表示用于第一次残差连接
		edge_feats_in1 = edge_feats  # 缓存边表示用于第一次残差连接
			
		# 在应用几何注意力之前应用第一轮归一化，以提高性能
		if self.apply_layer_norm:
			node_feats = self.layer_norm1_node_feats(node_feats)
			edge_feats = self.layer_norm1_edge_feats(edge_feats)
		else:  # 否则，默认使用批归一化
			node_feats = self.batch_norm1_node_feats(node_feats)
			edge_feats = self.batch_norm1_edge_feats(edge_feats)
		
		# 使用提供的节点和边表示获取多头注意力输出
		node_attn_out, edge_attn_out = self.mha_module(node_feats, edge_feats, edge_index)
		
		node_feats = node_attn_out.view(-1, self.num_output_feats)
		edge_feats = edge_attn_out.view(-1, self.num_output_feats)
		
		node_feats = F.dropout(node_feats, self.dropout_rate, training=self.training)
		edge_feats = F.dropout(edge_feats, self.dropout_rate, training=self.training)
		
		node_feats = self.O_node_feats(node_feats)
		edge_feats = self.O_edge_feats(edge_feats)
		
		# 进行第一次残差连接
		if self.residual:
			node_feats = node_feats_in1 + node_feats  # 进行第一次节点残差连接
			edge_feats = edge_feats_in1 + edge_feats  # 进行第一次边残差连接
		
		node_feats_in2 = node_feats  # 缓存节点表示用于第二次残差连接
		edge_feats_in2 = edge_feats  # 缓存边表示用于第二次残差连接
		
		# 在进行第一次残差连接后应用第二轮归一化
		if self.apply_layer_norm:
			node_feats = self.layer_norm2_node_feats(node_feats)
			edge_feats = self.layer_norm2_edge_feats(edge_feats)
		else:  # 否则，默认使用批归一化
			node_feats = self.batch_norm2_node_feats(node_feats)
			edge_feats = self.batch_norm2_edge_feats(edge_feats)
		
		# 对节点和边特征应用MLP
		for layer in self.node_feats_MLP:
			node_feats = layer(node_feats)
		for layer in self.edge_feats_MLP:
			edge_feats = layer(edge_feats)
		
		# 进行第二次残差连接
		if self.residual:
			node_feats = node_feats_in2 + node_feats  # 进行第二次节点残差连接
			edge_feats = edge_feats_in2 + edge_feats  # 进行第二次边残差连接
		
		# 返回边表示和节点表示（用于接口预测以外的任务）
		return node_feats, edge_feats
	
	def forward(self, edge_index, node_feats, edge_feats):
		"""执行几何变换器的前向传递以获取中间节点和边表示。"""
		node_feats, edge_feats = self.run_gt_layer(edge_index, node_feats, edge_feats)
		return node_feats, edge_feats


class FinalGraphTransformerModule(nn.Module):
	"""（最终层）使用自注意力组合节点和边表示的图变换器模块。"""	
	def __init__(self,
					num_hidden_channels,
					activ_fn=nn.SiLU(),
					residual=True,
					num_attention_heads=4,
					norm_to_apply='batch',
					dropout_rate=0.1,
					num_layers=4):
		super(FinalGraphTransformerModule, self).__init__()
		
		# 记录给定的参数
		self.activ_fn = activ_fn
		self.residual = residual
		self.num_attention_heads = num_attention_heads
		self.norm_to_apply = norm_to_apply
		self.dropout_rate = dropout_rate
		self.num_layers = num_layers
		
		# --------------------
		# 变换器模块
		# --------------------
		# 定义与几何变换器模块相关的所有模块
		self.apply_layer_norm = 'layer' in self.norm_to_apply.lower()
		
		self.num_hidden_channels, self.num_output_feats = num_hidden_channels, num_hidden_channels
		if self.apply_layer_norm:
			self.layer_norm1_node_feats = nn.LayerNorm(self.num_output_feats)
			self.layer_norm1_edge_feats = nn.LayerNorm(self.num_output_feats)
		else:  # 否则，默认使用批归一化
			self.batch_norm1_node_feats = nn.BatchNorm1d(self.num_output_feats)
			self.batch_norm1_edge_feats = nn.BatchNorm1d(self.num_output_feats)
		
		self.mha_module = MultiHeadAttentionLayer(
					self.num_hidden_channels,
					self.num_output_feats // self.num_attention_heads,
					self.num_attention_heads,
					self.num_hidden_channels != self.num_output_feats,  # 只有当Linear()必须改变大小时才使用偏置
					update_edge_feats=False)
		
		self.O_node_feats = nn.Linear(self.num_output_feats, self.num_output_feats)
		
		# 节点特征的MLP
		dropout = nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0.0 else nn.Identity()
		self.node_feats_MLP = nn.ModuleList([
					nn.Linear(self.num_output_feats, self.num_output_feats * 2, bias=False),
					self.activ_fn,
					dropout,
					nn.Linear(self.num_output_feats * 2, self.num_output_feats, bias=False)
					])
		
		if self.apply_layer_norm:
			self.layer_norm2_node_feats = nn.LayerNorm(self.num_output_feats)
		else:  # 否则，默认使用批归一化
			self.batch_norm2_node_feats = nn.BatchNorm1d(self.num_output_feats)
		
		self.reset_parameters()
	
	def reset_parameters(self):
		"""重新初始化可学习参数。"""
		scale = 2.0
		glorot_orthogonal(self.O_node_feats.weight, scale=scale)
		self.O_node_feats.bias.data.fill_(0)
		
		for layer in self.node_feats_MLP:
			if hasattr(layer, 'weight'):  # 跳过激活函数的初始化
				glorot_orthogonal(layer.weight, scale=scale)
		
		#glorot_orthogonal(self.conformation_module.weight, scale=scale)
	
	def run_gt_layer(self, edge_index, node_feats, edge_feats):
		"""使用多头注意力（MHA）模块执行几何注意力的前向传递。"""
		node_feats_in1 = node_feats  # 缓存节点表示用于第一次残差连接
		#edge_feats = self.conformation_module(edge_feats)
		
		# 在应用几何注意力之前应用第一轮归一化，以提高性能
		if self.apply_layer_norm:
			node_feats = self.layer_norm1_node_feats(node_feats)
			edge_feats = self.layer_norm1_edge_feats(edge_feats)
		else:  # 否则，默认使用批归一化
			node_feats = self.batch_norm1_node_feats(node_feats)
			edge_feats = self.batch_norm1_edge_feats(edge_feats)
		
		# 使用提供的节点和边表示获取多头注意力输出
		node_attn_out, _ = self.mha_module(node_feats, edge_feats, edge_index)
		node_feats = node_attn_out.view(-1, self.num_output_feats)		
		node_feats = F.dropout(node_feats, self.dropout_rate, training=self.training)
		node_feats = self.O_node_feats(node_feats)
		
		# 进行第一次残差连接
		if self.residual:
			node_feats = node_feats_in1 + node_feats  # 进行第一次节点残差连接
		
		node_feats_in2 = node_feats  # 缓存节点表示用于第二次残差连接
		
		# 在进行第一次残差连接后应用第二轮归一化
		if self.apply_layer_norm:
			node_feats = self.layer_norm2_node_feats(node_feats)
		else:  # 否则，默认使用批归一化
			node_feats = self.batch_norm2_node_feats(node_feats)
		
		# 对节点特征应用MLP
		for layer in self.node_feats_MLP:
			node_feats = layer(node_feats)
		
		# 进行第二次残差连接
		if self.residual:
			node_feats = node_feats_in2 + node_feats  # 进行第二次节点残差连接
		
		# 返回节点表示
		return node_feats
	
	def forward(self, edge_index, node_feats, edge_feats):
		"""执行几何变换器的前向传递以获取最终节点表示。"""
		node_feats = self.run_gt_layer(edge_index, node_feats, edge_feats)
		return node_feats


class GraghTransformer(nn.Module):
	"""图变换器
	"""
	def __init__(
									self,
										in_channels, 
										edge_features=10,
										num_hidden_channels=128,
										activ_fn=nn.SiLU(),
										transformer_residual=True,
										num_attention_heads=4,
										norm_to_apply='batch',
										dropout_rate=0.1,
										num_layers=4,
										**kwargs
										):
		super(GraghTransformer, self).__init__()
		
		# 初始化模型参数
		self.activ_fn = activ_fn
		self.transformer_residual = transformer_residual
		self.num_attention_heads = num_attention_heads
		self.norm_to_apply = norm_to_apply
		self.dropout_rate = dropout_rate
		self.num_layers = num_layers
		
		# --------------------
		# 初始化模块
		# --------------------
		# 定义与边和节点初始化相关的所有模块
		self.node_encoder = nn.Linear(in_channels, num_hidden_channels)
		self.edge_encoder = nn.Linear(edge_features, num_hidden_channels) 
		# --------------------
		# 变换器模块
		# --------------------
		# 定义与可变数量的几何变换器模块相关的所有模块
		num_intermediate_layers = max(0, num_layers - 1)
		gt_block_modules = [GraphTransformerModule(
										num_hidden_channels=num_hidden_channels,
										activ_fn=activ_fn,
										residual=transformer_residual,
										num_attention_heads=num_attention_heads,
										norm_to_apply=norm_to_apply,
										dropout_rate=dropout_rate,
										num_layers=num_layers) for _ in range(num_intermediate_layers)]
		if num_layers > 0:
			gt_block_modules.extend([
											FinalGraphTransformerModule(
													num_hidden_channels=num_hidden_channels,
													activ_fn=activ_fn,
													residual=transformer_residual,
													num_attention_heads=num_attention_heads,
													norm_to_apply=norm_to_apply,
													dropout_rate=dropout_rate,
													num_layers=num_layers)])
		self.gt_block = nn.ModuleList(gt_block_modules)
	
	def forward(self, node_s, edge_s, edge_index):		
		node_feats = self.node_encoder(node_s)
		edge_feats = self.edge_encoder(edge_s)
			
		# 对给定的节点和边特征应用指定数量的中间几何注意力层
		for gt_layer in self.gt_block[:-1]:
			node_feats, edge_feats = gt_layer(edge_index, node_feats, edge_feats)
		
		# 应用最终层通过合并当前节点和边表示来更新节点表示
		node_feats = self.gt_block[-1](edge_index, node_feats, edge_feats)
		#return node_feats
		return node_feats

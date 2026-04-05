#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   dataloader_obj.py
@Time    :   2022/12/12 21:19:27
@Author  :   Xujun Zhang
@Version :   1.0
@Contact :   
@License :   
@Desc    :   无
'''

# 这里放置导入库
import torch
from collections.abc import Mapping, Sequence
from typing import List, Optional, Union, List
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Batch, Dataset
from torch_geometric.data.data import BaseData



class PassNoneCollater:
    def __init__(self, follow_batch, exclude_keys):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch):
        batch = list(filter(lambda x:x is not None, batch))
        elem = batch[0]
        if isinstance(elem, BaseData):
            return Batch.from_data_list(batch, self.follow_batch,
                                        self.exclude_keys)
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self([data[key] for data in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self(s) for s in zip(*batch)]

        raise TypeError(f'DataLoader found invalid type: {type(elem)}')

    def collate(self, batch):  # Deprecated...
        return self(batch)


class PassNoneDataLoader(torch.utils.data.DataLoader):
    r"""一个数据加载器，用于将 :class:`torch_geometric.data.Dataset` 中的数据对象合并为小批量。
    数据对象可以是 :class:`~torch_geometric.data.Data` 或 :class:`~torch_geometric.data.HeteroData` 类型。

    参数:
        dataset (Dataset): 从中加载数据的数据集。
        batch_size (int, optional): 每个批次加载多少样本。
            (默认值: :obj:`1`)
        shuffle (bool, optional): 如果设置为 :obj:`True`，数据将在每个 epoch 重新洗牌。
            (默认值: :obj:`False`)
        follow_batch (List[str], optional): 为列表中的每个键创建分配批次向量。
            (默认值: :obj:`None`)
        exclude_keys (List[str], optional): 将排除列表中的每个键。
            (默认值: :obj:`None`)
        **kwargs (optional): :class:`torch.utils.data.DataLoader` 的其他参数。
    """
    def __init__(
        self,
        dataset: Union[Dataset, List[BaseData]],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):

        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=PassNoneCollater(follow_batch, exclude_keys),
            **kwargs,
        )


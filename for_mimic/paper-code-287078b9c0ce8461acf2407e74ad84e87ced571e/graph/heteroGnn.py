import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
import torch
from .dimensionReduction import DimensionalityReductionNet
from .dimensionIncrease import dimensionincrease
device = 'cuda:0'

class HeteroGraphNN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        self.dim_reduce = DimensionalityReductionNet()
        self.dim_increase = dimensionincrease()
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        # 先进行维度降低
        inputs = {k: self.dim_reduce(v) for k, v in inputs.items()}

        # 输入是节点的特征字典
        h = self.conv1(graph, inputs)
        # 应用非线性激活函数+第一层残差
        h = {k: F.relu(v + inputs[k]) for k, v in h.items()}
        # 保留第一层的输出用于第二层的残差连接
        inputs_hid = h
        h = self.conv2(graph, h)
        h = {k: v + inputs_hid[k] for k, v in h.items()}

        # 最后进行维度恢复
        h = {k: self.dim_increase(v) for k, v in h.items()}
        return h
import dgl
import torch as th
g1 = dgl.heterograph({('user', 'plays', 'game'): [(0, 0), (1, 0)]})

g1.nodes['user'].data['h1'] = th.tensor([[0.], [1.]])
# 为“user”类型的节点创建第二个图并设置特征属性
g2 = dgl.heterograph({('user', 'plays', 'game'): [(0, 0)]})
g2.nodes['user'].data['h1'] = th.tensor([[0.]])
# 批处理图
graphs = []
graphs.append(g1)
graphs.append(g2)

bg = dgl.batch_hetero(graphs)
print(bg)
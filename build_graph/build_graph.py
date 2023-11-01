import dgl
import torch as th
# 创建一个具有3种节点类型和3种边类型的异构图
graph_data = {
   ('user', 'click', 'item'): (th.tensor([0, 0,1]), th.tensor([0, 1,2])),
   ('user', 'like', 'item'): (th.tensor([0, 0, 1, 1]), th.tensor([1, 2, 0, 1])),
   ('item', 'be_click', 'user'): (th.tensor([0, 1,2]), th.tensor([0, 0,1])),
   ('item', 'be_like', 'user'): (th.tensor([1, 2, 0, 1]), th.tensor([0, 0, 1, 1])),
}
g = dgl.heterograph(graph_data)
print(g.ntypes)
print(g.etypes)
print(g.canonical_etypes)


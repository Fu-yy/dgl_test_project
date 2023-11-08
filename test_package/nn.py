import torch
import dgl
import torch
import dgl
import numpy as np
import torch as th
from dgl.nn.pytorch import GraphConv
from utils import print_graph
import dgl.nn.pytorch as dglnn


def test01():
    g = dgl.heterograph({
        ('user', 'follows', 'user'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
        ('user', 'follows', 'game'): (torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3])),
        ('user', 'plays', 'game'): (torch.tensor([1, 3]), torch.tensor([2, 3]))
    })

    test = g['plays']
    print_graph(test)
    print()
    if ('by' in g.canonical_etypes):
        print("a")
    else:
        print("b")
    # print(g['plays'].in_degrees().unsqueeze(1))
    print(g.in_degrees(etype='plays'))
    sumNum = 0
    for stype, etype, dtype in g.canonical_etypes:
        if (etype == None):
            continue
        elif (etype == 'follows'):
            sumNum += g[etype].in_degrees() * 2
        elif (etype == 'plays'):
            sumNum += g[etype].in_degrees() * 3
        else:
            continue

    print(sumNum)


# rel_graph = g[stype, etype, dtype]
#
# print(rel_graph.canonical_etypes)
#
# print(rel_graph.etypes)


def testGraphConv():
    g = dgl.heterograph({
        ('user', 'follows', 'user'): ([0, 1], [1, 2]),
        ('user', 'plays', 'game'): ([0], [1]),
        ('store', 'sells', 'game'): ([0], [2])})
    conv = dglnn.HeteroGraphConv({
        'follows': dglnn.GraphConv(2, 3),
        'plays': dglnn.GraphConv(2, 3),
        'sells': dglnn.GraphConv(2, 3)},
        aggregate='sum')

    g1 = g.number_of_nodes('user')
    h1 = {'user': th.ones((g.number_of_nodes('user'), 2)),
          'game': th.ones((g.number_of_nodes('game'), 2)),
          'store': th.ones((g.number_of_nodes('store'), 2))}
    print(h1)
    h2 = conv(g, h1)
    print(h2)
    print(h2.keys())


testGraphConv()
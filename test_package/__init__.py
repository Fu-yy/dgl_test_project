
import os
import torch
import dgl
import pandas as pd
from dgl import random, seed
from test_package.testScssion import SessionDataset
from utils import print_graph


def generate_graph(data):
    # 将数据转换为DataFrame
    df = pd.DataFrame(data, columns=['user_id', 'item_id', 'click', 'like', 'time'])

    # 从1开始编号
    df['user'] = range(1, len(df) + 1)
    df['item'] = range(1, len(df) + 1)

    # 构建异构图的边类型和源节点、目标节点的张量表示
    edges = {
        ('item', 'click_by', 'user'): (torch.tensor(df['item_id']), torch.tensor(df['user_id'])),
        ('user', 'click', 'item'): (torch.tensor(df['user_id']), torch.tensor(df['item_id'])),
        ('item', 'like_by', 'user'): (torch.tensor(df['item_id']), torch.tensor(df['user_id'])),
        ('user', 'like', 'item'): (torch.tensor(df['user_id']), torch.tensor(df['item_id']))
    }

    # 创建异构图
    graph = dgl.heterograph(edges)
    # 将时间属性从字符串转换为整数
    df['time'] = df['time'].astype(int)
    # 设置边的时间属性
    graph.edges['click_by'].data['time'] = torch.LongTensor(df['time'])
    graph.edges['click'].data['time'] = torch.LongTensor(df['time'])
    graph.edges['like_by'].data['time'] = torch.LongTensor(df['time'])
    graph.edges['like'].data['time'] = torch.LongTensor(df['time'])

    # 设置节点的属性和ID
    graph.nodes['user'].data['user_id_click'] = torch.LongTensor(df['user_id'])
    graph.nodes['user'].data['user_id_like'] = torch.LongTensor(df['user_id'])
    graph.nodes['item'].data['item_id_click'] = torch.LongTensor(df['item_id'])
    graph.nodes['item'].data['item_id_like'] = torch.LongTensor(df['item_id'])

    return graph


def load_csv_resort_id(data_path):

    '''


    :param data_path: 传入csv路径
    :return: 将user_id和item_id从0到n重排序，为了适配建图
    '''

    # 读取CSV文件，假设目标列名为 'column_name'
    df = pd.read_csv(data_path)

    # 获取目标列的唯一值列表
    unique_values_user = df['user_id'].unique()
    unique_values_item = df['item_id'].unique()

    # 对唯一值列表进行排序并创建值到索引的映射
    value_to_index_user = {value: index for index, value in enumerate(sorted(unique_values_user))}
    value_to_index_item = {value: index for index, value in enumerate(sorted(unique_values_item))}

    # 替换目标列中的值
    df['user_id'] = df['user_id'].map(value_to_index_user)
    df['item_id'] = df['item_id'].map(value_to_index_item)

    # 打印替换后的结果
    # print(df['user_id'])
    # print(df['item_id'])
    return df


def testGrapg():
    # 创建节点类型和边类型的列表
    node_types = ['user', 'item']
    edge_types = ['like', 'like_by', 'click', 'click_by']

    # 创建节点和边的列表
    user_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    item_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    like_edges = [(0, 1), (1, 2), (2, 3), (5, 6), (6, 7), (8, 9), (10, 11), (13, 14)]
    like_by_edges = [(1, 0), (2, 1), (3, 2), (6, 5), (7, 6), (9, 8), (11, 10), (14, 13)]
    click_edges = [(3, 4), (7, 8), (9, 10), (15, 16)]
    click_by_edges = [(4, 3), (8, 7), (10, 9), (16, 15)]

    # 创建异构图
    graph = dgl.heterograph({
        ('user', 'like', 'item'): like_edges,
        ('item', 'like_by', 'user'): like_by_edges,
        ('user', 'click', 'item'): click_edges,
        ('item', 'click_by', 'user'): click_by_edges
    })

    # 添加节点特征
    graph.nodes['user'].data['user_id'] = torch.tensor(user_ids)
    graph.nodes['item'].data['item_id'] = torch.tensor(item_ids)

    # 输出异构图信息
    print_graph(graph)


def testheteGraph():

    return


if __name__ == '__main__':

    print("a")


    # 测试数据
    # data = [
    #     [1, 1, 1, 0, '001'],
    #     [1, 2, 0, 1, '002'],
    #     [1, 3, 1, 1, '003'],
    #     [2, 1, 0, 0, '002'],
    #     [2, 1, 1, 0, '003']
    # ]

    # graph = generate_graph(data)
    # print(graph)
    # My_path = os.path.dirname(os.path.dirname(__file__))
    #
    # My_path.replace('/', '\\')
    #
    # data_path = os.path.join(My_path, 'Data', 'test.csv')
    # load_csv_resort_id(data_path)

    # testGrapg()
    # testCompact()

    # u_temp_click1 = torch.ones(2,3)
    # u_temp_click2 = 2*torch.ones(4,3)
    #
    # D = 2 * torch.ones(2, 4)
    #
    # print(torch.cat((u_temp_click1, u_temp_click2), 0).size())
    # print(torch.cat((u_temp_click1, D), -1))

    # u_temp_click1 = torch.tensor([1])
    # u_temp_click2 = torch.tensor([1])
    # list_test = [u_temp_click1,u_temp_click2]
    # testResult = torch.tensor(list_test).long()
    # print(testResult)








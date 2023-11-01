import sys

import dgl
import pandas as pd
import numpy as np
import datetime
import argparse
from dgl.sampling import sample_neighbors, select_topk
import torch
import os
from dgl import save_graphs
from joblib import Parallel, delayed


# 计算item序列的相对次序
def cal_order(data):
    '''
    函数用于计算data中物品序列的相对次序。它首先根据时间（'time'）列对data进行排序，
    采用稳定的归并排序算法（kind='mergesort'）。然后通过range(len(data))为每个物品分配一个相对次序，
    并将结果存储在新的列order中。最后返回包含相对次序的data。
    '''
    data = data.sort_values(['time'], kind='mergesort')
    data['order'] = range(len(data))

    return data


# 计算user序列的相对次序
def cal_u_order(data):
    '''
    用于计算data中用户序列的相对次序。它的操作类似于cal_order函数，
    首先根据时间（'time'）列对data进行排序，
    然后为每个用户分配一个相对次序，
    并将结果存储在新的列u_order中。最后返回包含相对次序的data。
    '''
    data = data.sort_values(['time'], kind='mergesort')
    data['u_order'] = range(len(data))

    return data


def refine_time(data):
    '''
    用于调整data中的时间顺序，以确保时间的递增性。它首先根据时间（'time'）列对data进行排序，
    采用稳定的归并排序算法。然后遍历时间序列，如果相邻的时间值相等或者逆序，
    就将后续时间值加上一个增量time_gap，并递增time_gap。
    最后更新data中的时间列为调整后的时间序列，并返回调整后的data。



    目的：处理时间顺序错误，还有时间相同的数据    用归并排序稳定且复杂度低

    '''
    data = data.sort_values(['time'], kind='mergesort')
    time_seq = data['time'].values
    time_gap = 1
    for i, da in enumerate(time_seq[0:-1]):
        if time_seq[i] == time_seq[i + 1] or time_seq[i] > time_seq[i + 1]:
            time_seq[i + 1] = time_seq[i + 1] + time_gap
            time_gap += 1
    data['time'] = time_seq

    return data


def generate_graph(data):
    # 对输入的数据按照用户ID进行分组，并通过调用refine_time函数对每个用户的数据进行处理，使得每个用户的交互记录按时间排序。最后通过reset_index(drop=True)重置索引。
    data = data.groupby('user_id').apply(refine_time).reset_index(drop=True)

    # 对处理后的数据再次按照用户ID进行分组，并通过调用cal_order函数为每个用户的交互记录计算相对次序。最后通过reset_index(drop=True)重置索引。 也就是根据时间排序然后加一列顺序索引号order
    data = data.groupby('user_id').apply(cal_order).reset_index(drop=True)

    # 对处理后的数据按照物品ID进行分组，并通过调用cal_u_order函数为每个物品的交互记录计算相对次序。最后通过reset_index(drop=True)重置索引。
    data = data.groupby('item_id').apply(cal_u_order).reset_index(drop=True)

    # 将处理后的数据中的用户ID列转换为NumPy数组。/
    user = data['user_id'].values
    # 将处理后的数据中的物品ID列转换为NumPy数组。
    item = data['item_id'].values

    # 将处理后的数据中的时间列转换为NumPy数组。
    time = data['time'].values

    # 将处理后的数据中的like和click列转换为NumPy数组。
    # click = data['click'].values
    #
    # like = data['like'].values



    # 是否可以加边，user--user  item--item
    # 构建了一个字典graph_data，其中包含了两种类型的边：由物品到用户的边（'item'到'user'）和由用户到物品的边（'user'到'item'）。每条边的值是一个元组，包含了源节点和目标节点的张量表示。
    graph_data = {('item', 'by', 'user'): (torch.tensor(item), torch.tensor(user)),
                  ('user', 'pby', 'item'): (torch.tensor(user), torch.tensor(item))}
    # 使用dgl.heterograph函数创建一个异构图（heterograph），根据graph_data中的边和节点信息构建图。
    graph = dgl.heterograph(graph_data)

    # print(graph.number_of_nodes('item'))

    # 将图中由用户到物品的边（'by'类型）的时间属性（'time'）设置为time的张量表示。
    graph.edges['by'].data['time'] = torch.LongTensor(time)

    # 将图中由物品到用户的边（'pby'类型）的时间属性（'time'）设置为time的张量表示
    graph.edges['pby'].data['time'] = torch.LongTensor(time)
    # graph.edges['by'].data['t'] = torch.tensor(data['order'])
    # graph.edges['by'].data['rt'] = torch.tensor(data['re_order'])
    # graph.edges['pby'].data['t'] = torch.tensor(data['u_order'])
    # graph.edges['pby'].data['rt'] = torch.tensor(data['u_re_order'])

    # 将图中用户节点的属性（'user_id'）设置为唯一的用户ID的张量表示。
    graph.nodes['user'].data['user_id'] = torch.LongTensor(np.unique(user))

    # print(graph.number_of_nodes('item'))

    # 将图中物品节点的属性（'item_id'）设置为唯一的物品ID的张量表示
    graph.nodes['item'].data['item_id'] = torch.LongTensor(np.unique(item))
    # graph.nodes['item'].data['last_user'] = torch.tensor(data['u_last'])
    # graph.nodes['user'].data['last_item'] = torch.tensor(data['last'])
    return graph


def generate_user(user, data, graph, item_max_length, user_max_length, train_path, test_path, val_path, k_hop=3):
    # 从data中选择与当前用户ID匹配的行，并按照时间列进行排序。将结果存储在data_user中  返回的是一个用户的list（猜想）
    data_user = data[data['user_id'] == user].sort_values('time')  # 分别处理每个用户id对应的各项数据

    # print(data_user.head(30))
    # print(data.head(30))
    # 从data_user中提取时间列的值，并存储在u_time中。
    u_time = data_user['time'].values
    # print(u_time)
    # 从data_user中提取物品ID列的值，并存储在u_seq中。
    u_seq = data_user['item_id'].values
    # print(u_seq)
    # 计算u_seq的长度减1，并将结果存储在split_point中。
    split_point = len(u_seq) - 1
    # print(split_point)
    # test_len_seq = len(u_seq);
    # print("test_len_seq")
    # print(test_len_seq)
    # 初始化train_num和test_num变量，用于统计生成的训练数据和测试数据的数量。
    train_num = 0
    test_num = 0
    # 生成训练数据

    # 如果u_seq的长度小于3，表示该用户的序列过短，无法生成有效的训练数据和测试数据，因此直接返回0。
    # if len(u_seq) < 0:
    if len(u_seq) < 3:
        return 0, 0
    else:
        #  对于u_time中索引从0到倒数第2个的每个元素，同时迭代对应的索引值j和时间值t。
        for j, t in enumerate(u_time[0:-1]):

            # 如果j为0，跳过当前循环，继续下一个循环。这是因为第一个时间点没有前一个时间点，无法生成有效的训练数据。
            if j == 0:
                continue

            # 如果j小于item_max_length，将start_t设置为u_time的第一个时间值。这是为了确保生成的子图中包含足够的历史时间范围。-----有问题
            if j < item_max_length:
                start_t = u_time[0]
            # 如果j大于等于item_max_length，将start_t设置为u_time中第 j - item_max_length 个时间值。这是为了确保生成的子图中包含足够的历史时间范围。
            else:
                start_t = u_time[j - item_max_length]

            # 根据给定的时间范围，选择图中与当前用户相关的边的索引。sub_u_eid是一个布尔数组，指示哪些边满足时间范围条件。
            sub_u_eid = (graph.edges['by'].data['time'] < u_time[j + 1]) & (graph.edges['by'].data['time'] >= start_t)
            # 根据给定的时间范围，选择图中与当前物品相关的边的索引。sub_i_eid是一个布尔数组，指示哪些边满足时间范围条件。
            sub_i_eid = (graph.edges['pby'].data['time'] < u_time[j + 1]) & (graph.edges['pby'].data['time'] >= start_t)
            # 根据选择的边构建子图sub_graph，这个子图包含了在给定时间范围内与当前用户和物品相关的边。----根据这些边构建子图
            sub_graph = dgl.edge_subgraph(graph, edges={'by': sub_u_eid, 'pby': sub_i_eid}, relabel_nodes=False)

            # node_types = sub_graph.ntypes
            # edge_types = sub_graph.etypes
            # print(" sub_graph Node data:")
            # for ntype in node_types:
            #     print(f"sub_graph Nodes of type '{ntype}':")
            #     print(sub_graph.nodes[ntype].data)
            #
            # print("sub_graph Edge data:")
            # for etype in edge_types:
            #     print(f"sub_graph Edges of type '{etype}':")
            #     print(sub_graph.edges[etype].data)

            # 将当前用户ID转换为张量u_temp。
            u_temp = torch.tensor([user])
            # 将当前用户ID转换为张量 his_user，用于存储历史用户ID。
            his_user = torch.tensor([user])

            # 根据时间权重和item_max_length选择与当前用户相关的前item_max_length个物品节点，构建子图graph_i。
            graph_i = select_topk(sub_graph, item_max_length, weight='time', nodes={'user': u_temp})

            '''
               dgl.sampling.select_topk(g, k, weight, nodes=None, edge_dir='in', ascending=False, copy_ndata=True, copy_edata=True)
                 dgl库中的一个函数，用于从图中选择与给定节点相关的具有 k 个最大或最小权重的相邻边，然后返回由这些边所引入的子图。

                以下是函数的参数和详细解释：

                g: 输入的原始图。
                k: 每个节点选择的边的数量，即要选择的边的排名前 k 的边。
                weight: 衡量边权重的标准。可以是边上的某个特征或属性，用于确定边的权重大小。
                nodes: 一个指定节点的列表，表示要从哪些节点出发选择边。如果为 None，则选择图中所有节点的边。
                edge_dir: 边的方向。可以是 'in'（入边）或 'out'（出边）。表示选择与节点入边或出边相关的权重。
                ascending: 一个布尔值，指定是否按升序（True）或降序（False）排列边的权重。如果为 True，将选择排名最低的权重。
                copy_ndata: 一个布尔值，指定是否将节点特征数据复制到子图中。
                copy_edata: 一个布尔值，指定是否将边特征数据复制到子图中。
            '''

            # node_types = graph_i.ntypes
            # edge_types = graph_i.etypes
            # print(" graph_i Node data:")
            # for ntype in node_types:
            #     print(f"graph_i Nodes of type '{ntype}':")
            #     print(graph_i.nodes[ntype].data)
            #
            # print("graph_i Edge data:")
            # for etype in edge_types:
            #     print(f"graph_i Edges of type '{etype}':")
            #     print(graph_i.edges[etype].data)

            # 从graph_i中获取物品节点的唯一标识，并存储在张量i_temp中，用于存储历史物品ID。

            i_temp = torch.unique(graph_i.edges(etype='by')[0])

            # 从graph_i中获取物品节点的唯一标识，并存储在张量his_item中，用于存储历史物品ID。
            his_item = torch.unique(graph_i.edges(etype='by')[0])

            #  从graph_i中获取边的全局ID，并将结果存储在名为edge_i的列表中。
            edge_i = [graph_i.edges['by'].data[dgl.NID]]

            # print(i_temp)
            # print(his_item)
            # print(edge_i)

            # 初始化名为edge_u的空列表，用于存储用户的边的全局ID。
            edge_u = []
            # 迭代k_hop-1次，用于生成多跳子图。
            for _ in range(k_hop - 1):
                # 根据时间权重和user_max_length选择与历史物品节点相关的前user_max_length个用户节点，构建子图graph_u。
                graph_u = select_topk(sub_graph, user_max_length, weight='time', nodes={'item': i_temp})  # item的邻居user

                # 从graph_u中获取用户节点的唯一标识，确保这些节点不在历史用户ID中，并选择最后user_max_length个节点。将结果存储在u_temp中。
                u_temp = np.setdiff1d(torch.unique(graph_u.edges(etype='pby')[0]), his_user)[-user_max_length:]
                # u_temp = torch.unique(torch.cat((u_temp, graph_u.edges(etype='pby')[0])))

                # 根据时间权重和item_max_length选择与历史用户节点相关的前item_max_length个物品节点，构建子图graph_i。
                graph_i = select_topk(sub_graph, item_max_length, weight='time', nodes={'user': u_temp})

                # 将u_temp和his_user中的用户ID合并，并去除重复项，更新his_user。
                his_user = torch.unique(torch.cat([torch.tensor(u_temp), his_user]))
                # i_temp = torch.unique(torch.cat((i_temp, graph_i.edges(etype='by')[0])))
                # 从graph_i中获取物品节点的唯一标识，确保这些节点不在历史物品ID中。将结果存储在i_temp中。
                i_temp = np.setdiff1d(torch.unique(graph_i.edges(etype='by')[0]), his_item)

                # 将i_temp和his_item中的物品ID合并，并去除重复项，更新his_item
                his_item = torch.unique(torch.cat([torch.tensor(i_temp), his_item]))

                # 从graph_i中获取边的全局ID，并将结果添加到edge_i列表中。
                edge_i.append(graph_i.edges['by'].data[dgl.NID])

                # 从graph_u中获取边的全局ID，并将结果添加到edge_u列表中。
                edge_u.append(graph_u.edges['pby'].data[dgl.NID])
            # 将edge_u列表中的边的全局ID合并，并去除重复项，存储在all_edge_u中。
            all_edge_u = torch.unique(torch.cat(edge_u))

            # 将edge_i列表中的边的全局ID合并，并去除重复项，存储在all_edge_i中。
            all_edge_i = torch.unique(torch.cat(edge_i))

            # 根据选择的边构建最终的子图fin_graph，该子图包含与历史用户和物品相关的边。
            fin_graph = dgl.edge_subgraph(sub_graph, edges={'by': all_edge_i, 'pby': all_edge_u})

            # 获取当前时间点的下一个物品ID，作为目标。
            target = u_seq[j + 1]

            # 获取当前时间点的物品ID，作为上一个物品。
            last_item = u_seq[j]

            testUser = fin_graph.nodes['user'].data['user_id'] == user
            testItem = fin_graph.nodes['item'].data['item_id'] == last_item
            # print(testUser)
            # print(testItem)

            # 从fin_graph中获取当前用户在节点中的索引，存储在u_alis中。
            u_alis = torch.where(fin_graph.nodes['user'].data['user_id'] == user)[0]

            # 从fin_graph中获取上一个物品在节点中的索引，存储在`last_alis
            last_alis = torch.where(fin_graph.nodes['item'].data['item_id'] == last_item)[0]
            # 分别计算user和last_item在fin_graph中的索引
            # print(u_alis)
            # print(last_alis)
            # 训练集
            if j < split_point - 1:
                save_graphs(train_path + '/' + str(user) + '/' + str(user) + '_' + str(j) + '.bin', fin_graph,
                            {'user': torch.tensor([user]), 'target': torch.tensor([target]), 'u_alis': u_alis,
                             'last_alis': last_alis})
                train_num += 1
            # 验证集
            if j == split_point - 1 - 1:
                save_graphs(val_path + '/' + str(user) + '/' + str(user) + '_' + str(j) + '.bin', fin_graph,
                            {'user': torch.tensor([user]), 'target': torch.tensor([target]), 'u_alis': u_alis,
                             'last_alis': last_alis})
            # 测试集
            if j == split_point - 1:
                save_graphs(test_path + '/' + str(user) + '/' + str(user) + '_' + str(j) + '.bin', fin_graph,
                            {'user': torch.tensor([user]), 'target': torch.tensor([target]), 'u_alis': u_alis,
                             'last_alis': last_alis})
                test_num += 1
        return train_num, test_num


def process_user(u, data, graph, item_max_length, user_max_length, train_path, test_path, val_path, k_hop):
    # 根据用户ID生成数据
    return generate_user(u, data, graph, item_max_length, user_max_length, train_path, test_path, val_path, k_hop)


def generate_data(data, graph, item_max_length, user_max_length, train_path, test_path, val_path, job=10, k_hop=3):
    '''
        data: 包含用户和物品信息的数据集。
        graph: 表示用户-物品关系的图。
        item_max_length: 物品序列的最大长度。
        user_max_length: 用户序列的最大长度。
        train_path: 训练集数据的保存路径。
        test_path: 测试集数据的保存路径。
        val_path: 验证集数据的保存路径。
        job: 并行处理的作业数，默认为10。
        k_hop: 图中的跳数，默认为3。

        results： 具体而言，results 的每个元素都是一个字典，字典的键值对如下：

            'user_id'：用户ID。
            'train_data'：该用户的训练数据，是一个列表，每个元素代表一个训练样本。
            'test_data'：该用户的测试数据，是一个列表，每个元素代表一个测试样本。
            其他可能的键值对，例如用于验证的数据、用户属性等。
            这个函数的返回值 a 是一个列表，列表的长度等于数据集中不同用户的数量，每个元素都是一个字典，代表一个用户的数据。
    '''
    # 获取所有用户的ID
    user = data['user_id'].unique()
    # print(user)
    # 使用普通的函数替代Lambda函数，并进行并行处理
    # n_jobs改成-1  需要的时候再改成10
    results = Parallel(n_jobs=-1)(
        delayed(process_user)
        (u, data, graph, item_max_length, user_max_length, train_path, test_path, val_path, k_hop)
        for u in user
    )
    # print("train_num, test_num")
    # print(results)
    # 处理每个任务的结果
    for result in results:
        # 在这里处理每个任务的结果
        print("result")
        print(result)
    # results
    return results


def mainLoadData(data, job, item_max_length, user_max_length, k_hop, result_path):
    # if __name__ == '__main__':

    opt = MyOptClass()

    opt.data = data
    opt.job = job
    opt.item_max_length = item_max_length
    opt.user_max_length = user_max_length
    opt.k_hop = k_hop
    opt.result_path = result_path

    # 这部分代码根据数据集名称构建数据路径和图路径。然后读取数据集文件（csv格式）并对用户ID进行分组，应用refine_time函数进行时间处理，并重置索引。最后将时间列转换为整数类型。

    # 添加绝对路径
    My_path = os.path.dirname(__file__)

    My_path.replace('/', '\\')

    data_path = os.path.join(My_path, 'Data', opt.data + '.csv')
    graph_path = os.path.join(My_path, 'Data', opt.data + '_graph.bin')
    data = pd.read_csv(data_path).groupby('user_id').apply(refine_time).reset_index(drop=True)  # 读取数据根据userid分组且根据时间排序
    data['time'] = data['time'].astype('int64')

    # print(data.head(30))
    # if opt.graph:
    #     graph = generate_graph(data)
    #     save_graphs(graph_path, graph)
    # else:

    # 这部分代码判断图数据是否已存在。如果图数据文件不存在，则调用generate_graph函数生成图数据，并使用save_graphs函数保存图数据到图路径。如果图数据文件已存在，则使用dgl.load_graphs函数加载图数据。
    if not os.path.exists(graph_path):
        graph = generate_graph(data)
        save_graphs(graph_path, graph)
    else:
        graph = dgl.load_graphs(graph_path)[0][0]

    # 这部分代码根据命令行参数创建训练集、验证集和测试集的存储路径。

    train_path = os.path.join(My_path, 'Newdata',
                              opt.data + '_' + str(opt.item_max_length) + '_' + str(opt.user_max_length) + '_' + str(
                                  opt.k_hop),
                              'train')
    val_path = os.path.join(My_path, 'Newdata',
                            opt.data + '_' + str(opt.item_max_length) + '_' + str(opt.user_max_length) + '_' + str(
                                opt.k_hop), 'val')
    test_path = os.path.join(My_path, 'Newdata',
                             opt.data + '_' + str(opt.item_max_length) + '_' + str(opt.user_max_length) + '_' + str(
                                 opt.k_hop), 'test')

    # 将标准输出重定向到文件--
    sys.stdout = open(result_path, 'w')

    print('start:', datetime.datetime.now())

    # 这部分代码调用generate_data函数，根据给定的data数据、graph图数据和参数生成训练集、验证集和测试集，并返回各个集合的数量。
    all_num = generate_data(data, graph, opt.item_max_length, opt.user_max_length, train_path, test_path, val_path,
                            job=opt.job, k_hop=opt.k_hop)
    # generate_data 传入一个迭代函数-->process_user-->generate_user()  主要这部分处理

    # 最大长度的选择通常依赖于具体任务和数据集的特点，可以通过尝试不同的最大长度值来进行调优。较大的最大长度可能会捕捉到更多的上下文信息，但也会增加计算和内存开销。较小的最大长度可能会丢失一部分信息。
    train_num = 0
    test_num = 0
    for num_ in all_num:
        train_num += num_[0]
        test_num += num_[1]
    print('The number of train set:', train_num)
    print('The number of test set:', test_num)
    print('end:', datetime.datetime.now())

    # 关闭文件和恢复标准输出
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    # return


class MyOptClass:
    def __init__(self, data='sample', graph=False, item_max_length=50, user_max_length=50, job=10, k_hop=2,
                 return_path=r"\default"):
        """
        构造函数

        参数:
        - data: 数据名称，默认值为 'sample'
        - graph: 是否使用图形，默认为 False
        - item_max_length: 最大物品长度，默认为 50
        - user_max_length: 最大用户长度，默认为 50
        - job: 训练的时期数，默认为 10
        - k_hop: k_hop，默认为 2
        """
        self.data = data
        self.graph = graph
        self.item_max_length = item_max_length
        self.user_max_length = user_max_length
        self.job = job
        self.k_hop = k_hop
        self.return_path = return_path

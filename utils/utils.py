#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/7/1 3:58
# @Author : ZM7
# @File : utils.py
# @Software: PyCharm
import os
from torch.utils.data import Dataset, DataLoader
import _pickle as cPickle
import dgl
import torch
import numpy as np
import pandas as pd

from utils import print_graph


def pickle_loader(path):
    a = cPickle.load(open(path, 'rb'))
    return a

def user_neg(data, item_num):

    '''
    这段代码定义了一个名为user_neg的函数，用于生成每个用户的负样本数据。

        函数的输入参数包括：

        data：一个包含用户-物品交互数据的DataFrame，其中至少包含'user_id'和'item_id'两列。
        item_num：一个整数，表示物品的总数量。
        函数的主要功能是根据输入的用户-物品交互数据，为每个用户生成负样本数据。具体步骤如下：

        item = range(item_num)：创建一个包含从0到(item_num-1)的整数范围的迭代器对象item，表示所有物品的标识符。

        select(data_u, item)是一个内部函数，用于从给定的用户交互数据data_u中选择负样本。它使用np.setdiff1d函数获取item和data_u之间的差集，即在item中存在但不在data_u中出现的物品标识符。这些物品标识符即为负样本。

        data.groupby('user_id')['item_id'].apply(lambda x: select(x, item))：首先，根据'user_id'列对data进行分组，然后对每个用户的'item_id'列应用select函数。这将为每个用户生成相应的负样本数据。最终的结果是一个Series对象，索引为用户ID，值为对应用户的负样本数据。

        函数返回生成的负样本数据。

        总体而言，该函数通过对用户-物品交互数据进行分组和差集操作，为每个用户生成负样本数据。负样本是指在物品总数量中存在但用户没有交互过的物品。生成的负样本数据以Series的形式返回，其中索引为用户ID，值为对应用户的负样本数据。
    '''
    item = range(item_num)
    def select(data_u, item):
        return np.setdiff1d(item, data_u)
    return data.groupby('user_id')['item_id'].apply(lambda x: select(x, item))

def neg_generate(user, data_neg, neg_num=100):

    '''

    函数的输入参数包括：

        user：一个包含用户ID的NumPy数组，表示用户集合。
        data_neg：一个字典或数组，存储了每个用户的负样本数据。
        neg_num：一个整数，表示每个用户需要生成的负样本数量，默认值为100。
    函数的主要功能是生成用户的负样本数据。具体步骤如下：

    创建一个形状为(len(user), neg_num)的全零NumPy数组neg，用于存储生成的负样本数据。数组的数据类型为np.int32，即32位整数。

    遍历user数组中的每个用户ID，使用enumerate函数获取索引i和对应的用户IDu。

    对于每个用户ID u，从data_neg[u]中随机选择neg_num个样本，且不可重复（replace=False），并将选择的样本存储到neg数组的第i行。

    完成遍历后，返回生成的负样本数组neg。

    总体而言，该函数根据输入的用户集合和负样本数据，为每个用户生成一定数量的负样本。负样本是通过随机选择每个用户的负样本数据集（data_neg）中的样本得到的。生成的负样本存储在一个NumPy数组中，并作为函数的输出返回。



    在机器学习和推荐系统中，负样本（Negative Samples）是指与已知的正样本（Positive Samples）不相似或不相关的样本。在推荐系统中，正样本通常表示用户已经喜欢或者有过行为交互的物品，而负样本则代表用户没有喜欢或者交互过的物品。

        负样本的作用是用于构建训练数据集，帮助模型学习区分用户的兴趣和喜好。通过将正样本与负样本进行对比，模型可以学习到正样本的特征和属性，并且区分出与之不相似的负样本。这有助于模型在实际应用中进行推荐时，能够更好地区分用户的兴趣，提供更准确的推荐结果。

        生成负样本的方式可以有多种，常见的方法包括：

        随机采样：从所有未被用户交互过的物品中随机选择一定数量的物品作为负样本。
        排序采样：根据物品的一些特征（如流行度、热度等），对未被用户交互过的物品进行排序，选择排名靠前的物品作为负样本。
        负例生成模型：使用生成模型（如生成对抗网络）来生成与已知正样本不相似的负样本。
        生成负样本的目的是为了训练模型能够准确地区分用户的喜好和兴趣，从而提高推荐系统的准确性和个性化程度。
    '''
    neg = np.zeros((len(user), neg_num), np.int32)
    for i, u in enumerate(user):
        neg[i] = np.random.choice(data_neg[u], neg_num, replace=False)
    return neg


class myFloder(Dataset):


    def __init__(self, root_dir, loader):
        self.root = root_dir
        self.loader = loader
        # loader可以通过路径加载图形数据
        self.dir_list = load_data(root_dir)  # 通过load_data（）加载root_dir的文件
        self.size = len(self.dir_list)

    def __getitem__(self, index):
        dir_ = self.dir_list[index]
        data = self.loader(dir_)# 可以通过路径加载图形数据

        return data

    def __len__(self):
        return self.size


def collate(data):
    user = []
    graph = []
    last_item = []
    label = []
    for da in data:
        user.append(da[0])
        graph.append(da[1])
        last_item.append(da[2])
        label.append(da[3])
    return torch.Tensor(user).long(), dgl.batch_hetero(graph), torch.Tensor(last_item).long(), torch.Tensor(label).long()


def load_data(data_path):
    data_dir = []
    dir_list = os.listdir(data_path)  #所有文件夹名称的列表
    dir_list.sort()
    for filename in dir_list:
        for fil in os.listdir(os.path.join(data_path, filename)):
            data_dir.append(os.path.join(os.path.join(data_path, filename), fil))
    return data_dir


def collate_test(data, user_neg):
    # 生成负样本和每个序列的长度
    user_alis = []
    graph = []
    last_item = []
    label = []
    user = []
    length = []
    for da in data:
        user_alis.append(da[0])
        graph.append(da[1])
        last_item.append(da[2])
        label.append(da[3])
        user.append(da[4])
        length.append(da[5])
    return torch.Tensor(user_alis).long(), dgl.batch_hetero(graph), torch.Tensor(last_item).long(), \
           torch.Tensor(label).long(), torch.Tensor(length).long(), torch.Tensor(neg_generate(user, user_neg)).long()


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def eval_metric(all_top, all_label, all_length, random_rank=True):
    recall5, recall10, recall20, ndgg5, ndgg10, ndgg20 = [], [], [], [], [], []
    data_l = np.zeros((100, 7))
    for index in range(len(all_top)):
        per_length = all_length[index]
        if random_rank:
            prediction = (-all_top[index]).argsort(1).argsort(1)
            predictions = prediction[:, 0]
            for i, rank in enumerate(predictions):
                # data_l[per_length[i], 6] += 1
                if rank < 20:
                    ndgg20.append(1 / np.log2(rank + 2))
                    recall20.append(1)
                    # if per_length[i]-1 < 100:
                    #     data_l[per_length[i], 5] += 1 / np.log2(rank + 2)
                    #     data_l[per_length[i], 2] += 1
                    # else:
                    #     data_l[99, 5] += 1 / np.log2(rank + 2)
                    #     data_l[99, 2] += 1
                else:
                    ndgg20.append(0)
                    recall20.append(0)
                if rank < 10:
                    ndgg10.append(1 / np.log2(rank + 2))
                    recall10.append(1)
                    # if per_length[i]-1 < 100:
                    #     data_l[per_length[i], 4] += 1 / np.log2(rank + 2)
                    #     data_l[per_length[i], 1] += 1
                    # else:
                    #     data_l[99, 4] += 1 / np.log2(rank + 2)
                    #     data_l[99, 1] += 1
                else:
                    ndgg10.append(0)
                    recall10.append(0)
                if rank < 5:
                    ndgg5.append(1 / np.log2(rank + 2))
                    recall5.append(1)
                    # if per_length[i]-1 < 100:
                    #     data_l[per_length[i], 3] += 1 / np.log2(rank + 2)
                    #     data_l[per_length[i], 0] += 1
                    # else:
                    #     data_l[99, 3] += 1 / np.log2(rank + 2)
                    #     data_l[99, 0] += 1
                else:
                    ndgg5.append(0)
                    recall5.append(0)

        else:
            for top_, target in zip(all_top[index], all_label[index]):
                recall20.append(np.isin(target, top_))
                recall10.append(np.isin(target, top_[0:10]))
                recall5.append(np.isin(target, top_[0:5]))
                if len(np.where(top_ == target)[0]) == 0:
                    ndgg20.append(0)
                else:
                    ndgg20.append(1 / np.log2(np.where(top_ == target)[0][0] + 2))
                if len(np.where(top_ == target)[0]) == 0:
                    ndgg10.append(0)
                else:
                    ndgg10.append(1 / np.log2(np.where(top_ == target)[0][0] + 2))
                if len(np.where(top_ == target)[0]) == 0:
                    ndgg5.append(0)
                else:
                    ndgg5.append(1 / np.log2(np.where(top_ == target)[0][0] + 2))
    #pd.DataFrame(data_l, columns=['r5','r10','r20','n5','n10','n10','number']).to_csv(name+'.csv')
    return np.mean(recall5), np.mean(recall10), np.mean(recall20), np.mean(ndgg5), np.mean(ndgg10), np.mean(ndgg20), \
           pd.DataFrame(data_l, columns=['r5','r10','r20','n5','n10','n20','number'])



def format_arg_str(args, exclude_lst, max_len=20):
    linesep = os.linesep
    arg_dict = vars(args)
    keys = [k for k in arg_dict.keys() if k not in exclude_lst]
    values = [arg_dict[k] for k in keys]
    key_title, value_title = 'Arguments', 'Values'
    key_max_len = max(map(lambda x: len(str(x)), keys))
    value_max_len = min(max(map(lambda x: len(str(x)), values)), max_len)
    key_max_len, value_max_len = max([len(key_title), key_max_len]), max([len(value_title), value_max_len])
    horizon_len = key_max_len + value_max_len + 5
    res_str = linesep + '=' * horizon_len + linesep
    res_str += ' ' + key_title + ' ' * (key_max_len - len(key_title)) + ' | ' \
               + value_title + ' ' * (value_max_len - len(value_title)) + ' ' + linesep + '=' * horizon_len + linesep
    for key in sorted(keys):
        value = arg_dict[key]
        if value is not None:
            key, value = str(key), str(value).replace('\t', '\\t')
            value = value[:max_len-3] + '...' if len(value) > max_len else value
            res_str += ' ' + key + ' ' * (key_max_len - len(key)) + ' | ' \
                       + value + ' ' * (value_max_len - len(value)) + linesep
    res_str += '=' * horizon_len
    return res_str
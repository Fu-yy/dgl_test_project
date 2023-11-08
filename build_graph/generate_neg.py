#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/1/31 11:15
# @Author : ZM7
# @File : generate_neg
# @Software: PyCharm
import pandas as pd
import pickle
from utils import myFloder, pickle_loader, collate, trans_to_cuda, eval_metric, collate_test, user_neg
import os

'''
这段代码针对名为"Games"的数据集进行处理，并包括以下步骤：

dataset = 'Games'：将数据集名称赋值给变量dataset。

data = pd.read_csv('./Data/' + dataset + '.csv')：从文件路径'./Data/Games.csv'读取CSV格式的数据，并将其存储在名为data的DataFrame中。

user = data['user_id'].unique()：提取data中唯一的用户ID，将其存储在名为user的NumPy数组中。

item = data['item_id'].unique()：提取data中唯一的物品ID，将其存储在名为item的NumPy数组中。

user_num = len(user)：计算user数组的长度，即用户的数量，并将结果存储在变量user_num中。

item_num = len(item)：计算item数组的长度，即物品的数量，并将结果存储在变量item_num中。

data_neg = user_neg(data, item_num)：调用名为user_neg的函数，传递data和item_num作为参数，生成负样本数据集data_neg。函数的具体功能和实现细节无法确定，因为在提供的代码片段中没有包含该函数的定义。

f = open(dataset+'_neg', 'wb')：使用二进制模式创建一个文件对象f，文件名为'Games_neg'，用于存储负样本数据。

pickle.dump(data_neg,f)：使用pickle模块将负样本数据data_neg写入文件对象f中，持久化保存。

f.close()：关闭文件对象f。

'''


def gen_neg(dataset):
    # 获取当前文件的目录
    project_dir = os.path.dirname(os.path.dirname(__file__))

    dataset = 'test'
    file_path = os.path.join(project_dir, 'Data', dataset + '.csv')
    file_path = file_path.replace('/', '\\')

    data = pd.read_csv(file_path)
    user = data['user_id'].unique()
    item = data['item_id'].unique()
    user_num = len(user)
    item_num = len(item)

    data_neg = user_neg(data, item_num)

    with open(os.path.join(project_dir,'Data', dataset + '_neg'), 'wb') as f:
        pickle.dump(data_neg, f)

#
# if __name__ == '__main__':
#     gen_neg('cd')

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/11/17 4:32
# @Author : ZM7
# @File : new_main
# @Software: PyCharm

import datetime
from functools import partial

import torch
from sys import exit
import pandas as pd
import numpy as np
import dgl

from model import DGSR, collate, neg_generate, collate_fn_test, collate_test,eval_metric, mkdir_if_not_exist, Logger
from dgl import load_graphs
import pickle
from utils import myFloder
import warnings
import argparse
import os
import sys
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn




class OptClass:
    def __init__(self, data='sample', batchSize=50, hidden_size=50, epoch=10, lr=0.001, l2=0.0001, user_update='rnn',
                 item_update='rnn', user_long='orgat', item_long='orgat', user_short='att', item_short='att',
                 feat_drop=0.3, attn_drop=0.3, layer_num=3, item_max_length=50, user_max_length=50, k_hop=2, gpu='4',
                 last_item=False, record=False, val=False, model_record=False):
        """
        Configuration class for your model.

        Args:
            data (str): Data name. Default is 'sample'.
            batchSize (int): Input batch size. Default is 50.
            hidden_size (int): Hidden state size. Default is 50.
            epoch (int): Number of epochs to train for. Default is 10.
            lr (float): Learning rate. Default is 0.001.
            l2 (float): L2 penalty. Default is 0.0001.
            user_update (str): User update method. Default is 'rnn'.
            item_update (str): Item update method. Default is 'rnn'.
            user_long (str): User long-term aggregation method. Default is 'orgat'.
            item_long (str): Item long-term aggregation method. Default is 'orgat'.
            user_short (str): User short-term aggregation method. Default is 'att'.
            item_short (str): Item short-term aggregation method. Default is 'att'.
            feat_drop (float): Dropout for feature. Default is 0.3.
            attn_drop (float): Dropout for attention. Default is 0.3.
            layer_num (int): Number of GNN layers. Default is 3.
            item_max_length (int): Maximum length of item sequence. Default is 50.
            user_max_length (int): Maximum length of user sequence. Default is 50.
            k_hop (int): Sub-graph size. Default is 2.
            gpu (str): GPU id. Default is '4'.
            last_item (bool): Aggregate last item. Default is False.
            record (bool): Record experimental results. Default is False.
            val (bool): Use validation set. Default is False.
            model_record (bool): Record model. Default is False.
        """
        self.data = data
        self.batchSize = batchSize
        self.hidden_size = hidden_size
        self.epoch = epoch
        self.lr = lr
        self.l2 = l2
        self.user_update = user_update
        self.item_update = item_update
        self.user_long = user_long
        self.item_long = item_long
        self.user_short = user_short
        self.item_short = item_short
        self.feat_drop = feat_drop
        self.attn_drop = attn_drop
        self.layer_num = layer_num
        self.item_max_length = item_max_length
        self.user_max_length = user_max_length
        self.k_hop = k_hop
        self.gpu = gpu
        self.last_item = last_item
        self.record = record
        self.val = val
        self.model_record = model_record





if __name__ == '__main__':

    a = lambda x: collate_test(x, data_neg)



    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='sample', help='data name: sample')
    parser.add_argument('--batchSize', type=int, default=50, help='input batch size')
    parser.add_argument('--hidden_size', type=int, default=50, help='hidden state size')
    parser.add_argument('--epoch', type=int, default=10, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--l2', type=float, default=0.0001, help='l2 penalty')
    parser.add_argument('--user_update', default='rnn')
    parser.add_argument('--item_update', default='rnn')
    parser.add_argument('--user_long', default='orgat')
    parser.add_argument('--item_long', default='orgat')
    parser.add_argument('--user_short', default='att')
    parser.add_argument('--item_short', default='att')
    parser.add_argument('--feat_drop', type=float, default=0.3, help='drop_out')
    parser.add_argument('--attn_drop', type=float, default=0.3, help='drop_out')
    parser.add_argument('--layer_num', type=int, default=3, help='GNN layer')
    parser.add_argument('--item_max_length', type=int, default=50, help='the max length of item sequence')
    parser.add_argument('--user_max_length', type=int, default=50, help='the max length of use sequence')
    parser.add_argument('--k_hop', type=int, default=2, help='sub-graph size')
    parser.add_argument('--gpu', default='4')
    parser.add_argument('--last_item', action='store_true', help='aggreate last item')
    parser.add_argument("--record", action='store_true', default=False, help='record experimental results')
    parser.add_argument("--val", action='store_true', default=False)
    parser.add_argument("--model_record", action='store_true', default=False, help='record model')

    # opt = parser.parse_args()
    opt = OptClass()
    # 设置属性的值
    opt.data = 'test'
    opt.gpu = 0
    opt.epoch = 20
    opt.hidden_size = 50
    opt.batchSize = 5
    opt.user_long = 'orgat'
    opt.user_short = 'att'
    opt.item_long = 'orgat'
    opt.item_short = 'att'
    opt.user_update = 'rnn'
    opt.item_update = 'rnn'
    opt.lr = 0.001
    opt.l2 = 0.0001
    opt.layer_num = 2
    opt.item_max_length = 50
    opt.user_max_length = 50
    opt.attn_drop = 0.3
    opt.feat_drop = 0.3
    opt.record = True
    opt.k_hop = 3

    # 获取当前文件的目录
    project_dir = os.path.dirname(__file__)
    project_dir = project_dir.replace('/', '\\')

    # 拼接文件路径
    # file_path = os.path.join(current_dir, 'Data', 'test.txt')
    # 将斜杠替换为反斜杠

    args, extras = parser.parse_known_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
    device = torch.device('cuda:0')
    # print(opt)

    if opt.record:
        log_file = f'{project_dir}/results/{opt.data}_ba_{opt.batchSize}_G_{opt.gpu}_dim_{opt.hidden_size}_ulong_{opt.user_long}_ilong_{opt.item_long}_' \
                   f'US_{opt.user_short}_IS_{opt.item_short}_La_{args.last_item}_UM_{opt.user_max_length}_IM_{opt.item_max_length}_K_{opt.k_hop}' \
                   f'_layer_{opt.layer_num}_l2_{opt.l2}'
        mkdir_if_not_exist(log_file)
        sys.stdout = Logger(log_file)
        print(f'Logging to {log_file}')
    if opt.model_record:
        model_file = f'{project_dir}/{opt.data}_ba_{opt.batchSize}_G_{opt.gpu}_dim_{opt.hidden_size}_ulong_{opt.user_long}_ilong_{opt.item_long}_' \
                     f'US_{opt.user_short}_IS_{opt.item_short}_La_{args.last_item}_UM_{opt.user_max_length}_IM_{opt.item_max_length}_K_{opt.k_hop}' \
                     f'_layer_{opt.layer_num}_l2_{opt.l2}'

    # loading data
    data = pd.read_csv(os.path.join(project_dir, 'Data', opt.data + '.csv'))
    user = data['user_id'].unique()
    item = data['item_id'].unique()
    user_num = len(user)
    item_num = len(item)
    train_root = os.path.join(project_dir, 'Newdata',
                              opt.data + '_' + str(opt.item_max_length) + '_' + str(opt.user_max_length) + '_' + str(
                                  opt.k_hop), 'train')
    test_root = os.path.join(project_dir, 'Newdata',
                             opt.data + '_' + str(opt.item_max_length) + '_' + str(opt.user_max_length) + '_' + str(
                                 opt.k_hop), 'test')

    val_root = os.path.join(project_dir, 'Newdata',
                            opt.data + '_' + str(opt.item_max_length) + '_' + str(opt.user_max_length) + '_' + str(
                                opt.k_hop), 'val')
    train_set = myFloder(train_root, load_graphs)
    test_set = myFloder(test_root, load_graphs)#load_graphs为dgl i提供的加载图形数据的方法
    if opt.val:
        val_set = myFloder(val_root, load_graphs)

    print('train number:', train_set.size)
    print('test number:', test_set.size)
    print('user number:', user_num)
    print('item number:', item_num)
    # f = open(opt.data+'_neg', 'rb')
    # data_neg = pickle.load(f) # 用于评估测试集
    with open(os.path.join(project_dir,'Data',opt.data+'_neg'), 'rb') as f:
        data_neg = pickle.load(f)



    # 使用functools.partial()包装collate_fn函数，并传入额外的参数
    test_collate_fn = partial(collate_test,data=test_set, user_neg=data_neg)
    if opt.val:
        val_collate_fn = partial(collate_test,data=val_set, user_neg=data_neg)

    train_data = DataLoader(dataset=train_set, batch_size=opt.batchSize, collate_fn=collate, shuffle=True,
                            pin_memory=True, num_workers=12)
    test_data = DataLoader(dataset=test_set, batch_size=opt.batchSize, collate_fn=test_collate_fn, pin_memory=True,
                           num_workers=8)

    if opt.val:
        # 取消lambda
        # val_data = DataLoader(dataset=val_set, batch_size=opt.batchSize, collate_fn=lambda x: collate_test(x, data_neg), pin_memory=True, num_workers=2)
        val_data = DataLoader(dataset=val_set, batch_size=opt.batchSize, collate_fn=val_collate_fn, pin_memory=True,
                              num_workers=2)
    # 初始化模型
    model = DGSR(user_num=user_num, item_num=item_num, input_dim=opt.hidden_size, item_max_length=opt.item_max_length,
                 user_max_length=opt.user_max_length, feat_drop=opt.feat_drop, attn_drop=opt.attn_drop,
                 user_long=opt.user_long, user_short=opt.user_short,
                 item_long=opt.item_long, item_short=opt.item_short, user_update=opt.user_update,
                 item_update=opt.item_update, last_item=opt.last_item,
                 layer_num=opt.layer_num).cuda()

    print(model)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
    loss_func = nn.CrossEntropyLoss()
    best_result = [0, 0, 0, 0, 0, 0]  # hit5,hit10,hit20,mrr5,mrr10,mrr20
    best_epoch = [0, 0, 0, 0, 0, 0]
    stop_num = 0
    for epoch in range(opt.epoch):
        stop = True
        epoch_loss = 0
        iter = 0
        print('start training: ', datetime.datetime.now())
        model.train()
        for user, batch_graph, label, last_item in train_data: # train_data传入的是myFloder格式的DataLoader  而myFloader中直接处理了这个文件夹内容，直接读取了内容
            # user是tensor格式   user
            # print(user)
            # # print(batch_graph)
            # # batch_graph是DGLHeteroGraph图数据   graph  dgl.batch(graph),
            # print(label)
            # # label是tensor格式   target
            # print(last_item)
            # # last_item是tensor格式   last_alis
            #




            iter += 1
            score = model(batch_graph.to(device), user.to(device), last_item.to(device), is_training=True)
            loss = loss_func(score, label.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if iter % 400 == 0:
                print('Iter {}, loss {:.4f}'.format(iter, epoch_loss / iter), datetime.datetime.now())
        epoch_loss /= iter
        model.eval()
        print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss), '=============================================')

        # val
        if opt.val:
            print('start validation: ', datetime.datetime.now())
            val_loss_all, top_val = [], []
            with torch.no_grad:
                for user, batch_graph, label, last_item, neg_tar in val_data:
                    score, top = model(batch_graph.to(device), user.to(device), last_item.to(device),
                                       neg_tar=torch.cat([label.unsqueeze(1), neg_tar], -1).to(device),
                                       is_training=False)
                    val_loss = loss_func(score, label.cuda())
                    val_loss_all.append(val_loss.append(val_loss.item()))
                    top_val.append(top.detach().cpu().numpy())
                recall5, recall10, recall20, ndgg5, ndgg10, ndgg20 = eval_metric(top_val)
                print('train_loss:%.4f\tval_loss:%.4f\tRecall@5:%.4f\tRecall@10:%.4f\tRecall@20:%.4f\tNDGG@5:%.4f'
                      '\tNDGG10@10:%.4f\tNDGG@20:%.4f' %
                      (epoch_loss, np.mean(val_loss_all), recall5, recall10, recall20, ndgg5, ndgg10, ndgg20))

        # test
        print('start predicting: ', datetime.datetime.now())
        all_top, all_label, all_length = [], [], []
        iter = 0
        all_loss = []
        with torch.no_grad():
            for user, batch_graph, label, last_item, neg_tar in test_data:
                iter += 1
                score, top = model(batch_graph.to(device), user.to(device), last_item.to(device),
                                   neg_tar=torch.cat([label.unsqueeze(1), neg_tar], -1).to(device), is_training=False)
                test_loss = loss_func(score, label.cuda())
                all_loss.append(test_loss.item())
                all_top.append(top.detach().cpu().numpy())
                all_label.append(label.numpy())
                if iter % 200 == 0:
                    print('Iter {}, test_loss {:.4f}'.format(iter, np.mean(all_loss)), datetime.datetime.now())
            recall5, recall10, recall20, ndgg5, ndgg10, ndgg20 = eval_metric(all_top)
            if recall5 > best_result[0]:
                best_result[0] = recall5
                best_epoch[0] = epoch
                stop = False
            if recall10 > best_result[1]:
                if opt.model_record:
                    torch.save(model.state_dict(), 'save_models/' + model_file + '.pkl')
                best_result[1] = recall10
                best_epoch[1] = epoch
                stop = False
            if recall20 > best_result[2]:
                best_result[2] = recall20
                best_epoch[2] = epoch
                stop = False
                # ------select Mrr------------------
            if ndgg5 > best_result[3]:
                best_result[3] = ndgg5
                best_epoch[3] = epoch
                stop = False
            if ndgg10 > best_result[4]:
                best_result[4] = ndgg10
                best_epoch[4] = epoch
                stop = False
            if ndgg20 > best_result[5]:
                best_result[5] = ndgg20
                best_epoch[5] = epoch
                stop = False
            if stop:
                stop_num += 1
            else:
                stop_num = 0
            print('train_loss:%.4f\ttest_loss:%.4f\tRecall@5:%.4f\tRecall@10:%.4f\tRecall@20:%.4f\tNDGG@5:%.4f'
                  '\tNDGG10@10:%.4f\tNDGG@20:%.4f\tEpoch:%d,%d,%d,%d,%d,%d' %
                  (epoch_loss, np.mean(all_loss), best_result[0], best_result[1], best_result[2], best_result[3],
                   best_result[4], best_result[5], best_epoch[0], best_epoch[1],
                   best_epoch[2], best_epoch[3], best_epoch[4], best_epoch[5]))




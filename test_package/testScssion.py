import os
import pickle

import dgl
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import configparser

# 指定配置文件
from utils import Configurator

config_file = 'basic.ini'

# 从指定文件加载配置
conf = Configurator(config_file)

def load_data(dataset, data_path):
    """
    加载数据集。

    参数:
        dataset: 数据集名称。
        data_path: 数据路径。

    返回值:
        train_data: 训练数据。
        test_data: 测试数据。
        max_vid: 最大视频ID。
        max_uid: 最大用户ID。
    """
    # 检查是否已存在序列化的数据文件

    FILE_PATH = os.path.dirname(__file__)
    FILE_PATH = FILE_PATH.replace('/', '\\')
    data_path = os.path.join(FILE_PATH,'dataset')
    if not os.path.exists(os.path.join(data_path, dataset,'train_seq.pkl')):
        # 创建临时文件路径以保存数据
        print('try to build ', os.path.join(data_path, dataset,'train_seq.pkl') )
        with open(os.path.join(data_path, dataset,'train.pkl') , 'rb') as f:
            train_data = pickle.load(f)
        max_vid = 0
        max_uid = 0
        for u in train_data:
            if u > max_uid:
                max_uid = u
            for sess in train_data[u]:
                if max_vid < max(sess):
                    max_vid = max(sess)

        try:
            with open(os.path.join(data_path, dataset, 'all_test.pkl') , 'rb') as f:
                test_data = pickle.load(f)
        except:
            with open(os.path.join(data_path, dataset,'test.pkl')  , 'rb') as f:
                test_data = pickle.load(f)

        train_data = common_seq(train_data)  # 提取训练数据的常见序列
        test_data = common_seq(test_data)  # 提取测试数据的常见序列

        with open(os.path.join(data_path, dataset,'test_seq.pkl') , 'wb') as f:
            pickle.dump(test_data, f)  # 将处理后的测试数据序列化保存

        with open(os.path.join(data_path, dataset,'train_seq.pkl') , 'wb') as f:
            pickle.dump(train_data, f)  # 将处理后的训练数据序列化保存

        return train_data, test_data, max_vid, max_uid

    with open(os.path.join(data_path, dataset,'train_seq.pkl')  , 'rb') as f:
        train_data = pickle.load(f)
    max_vid = 0
    max_uid = 0
    for data in train_data:
        if data[0] > max_uid:
            max_uid = data[0]
        if max_vid < max(data[1]):
            max_vid = max(data[1])
        if max_vid < max(data[2]):
            max_vid = max(data[2])

    with open(os.path.join(data_path, dataset, 'test_seq.pkl') , 'rb') as f:
        test_data = pickle.load(f)
    for data in test_data:
        if data[0] > max_uid:
            max_uid = data[0]
        if max_vid < max(data[1]):
            max_vid = max(data[1])
        if max_vid < max(data[2]):
            max_vid = max(data[2])

    return train_data, test_data, max_vid, max_uid

def common_seq(data_list):
    """
    提取常见的序列数据。

    参数:
        data_list: 数据列表，包含用户和对应的序列。

    返回值:
        final_seqs: 最终的序列列表，每个序列包含用户ID、输入序列和标签。
    """
    out_seqs = []  # 输出的序列列表，每个序列去掉最后一个元素
    label = []  # 序列的标签列表，每个标签是序列的最后一个元素
    uid = []  # 用户ID列表

    for u in tqdm(data_list, desc='gen_seq...', leave=False):
        u_seqs = data_list[u]  # 获取用户对应的序列
        for seq in u_seqs:
            for i in range(1, len(seq)):
                uid.append(int(u))  # 将用户ID添加到列表中
                out_seqs.append(seq[:-i])  # 将去掉最后一个元素的序列添加到列表中
                label.append([seq[-i]])  # 将最后一个元素作为标签添加到列表中

    final_seqs = []
    for i in range(len(uid)):
        final_seqs.append([uid[i], out_seqs[i], label[i]])  # 将用户ID、输入序列和标签组合成一个序列

    return final_seqs


class SessionDataset(Dataset):
    def __init__(self,root_dir,loader, data, config, max_len=None):
        """
        会话数据集类，用于处理数据集的加载和预处理。

        参数:
            data: 数据列表，包含用户ID、浏览序列、标签等信息的元组。
            config: 配置信息字典。
            max_len: 序列的最大长度。

        属性:
            data: 数据列表，包含用户ID、浏览序列、标签等信息的元组。
            max_seq_len: 序列的最大长度。

        """
        super(SessionDataset, self).__init__()
        self.root = root_dir
        self.loader = loader
        self.data = data
        self.dir_list = load_data(root_dir)  # 通过load_data（）加载root_dir的文件

        if max_len:
            self.max_seq_len = max_len
        else:
            self.max_seq_len = config['dataset.seq_len']

    def __len__(self):
        """
        返回数据集的长度。

        返回值:
            数据集的长度。
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        根据索引获取数据集中的样本。

        参数:
            index: 样本索引。

            在此处定义了uid, browsed_ids, mask, seq_len, label, pos_idx

        返回值:
            uid: 用户ID。
            browsed_ids: 浏览的视频ID序列。
            mask: 序列的掩码，用于指示有效元素。
            seq_len: 序列的实际长度。
            label: 标签。
            pos_idx: 正样本的索引。



            要这些  uid, browsed_ids, mask, seq_len, label, pos_idx

        """
        dir_ = self.dir_list[index]
        data = self.loader(dir_)  # 可以通过路径加载图形数据

        user = []
        graph = []
        last_item = []
        label = []
        for da in data:
            user.append(da[0])
            graph.append(da[1])
            last_item.append(da[2])
            label.append(da[3])


        ###########################################
        # uid = np.array([data[0]], dtype=np.int)  # 用户ID
        # browsed_ids = np.zeros((self.max_seq_len), dtype=np.int)  # 浏览的视频ID序列
        # seq_len = len(data[1][-self.max_seq_len:])  # 序列的实际长度
        # mask = np.array([1 for i in range(seq_len)] + [0 for i in range(self.max_seq_len - seq_len)], dtype=np.int)  # 序列的掩码，用于指示有效元素
        # pos_idx = np.array([seq_len - i - 1 for i in range(seq_len)] + [0 for i in range(self.max_seq_len - seq_len)], dtype=np.int)  # 正样本的索引
        # browsed_ids[:seq_len] = np.array(data[1][-self.max_seq_len:])  # 将浏览的视频ID序列填充到对应位置
        # seq_len = np.array(seq_len, dtype=np.int)  # 序列的实际长度
        # label = np.array(data[2], dtype=np.int)  # 标签



        uid = user
        browsed_ids = last_item
        return data,uid,graph, browsed_ids, mask, seq_len, label, pos_idx


if __name__ == '__main__':
    SEQ_LEN = 10
    # dataset.name=lastfm
    # dataset.path=dataset
    train_data_ater, test_data, max_vid, max_uid = load_data(conf['dataset.name'], conf['dataset.path'])

    train_data = SessionDataset(train_data_ater, conf, max_len=SEQ_LEN)

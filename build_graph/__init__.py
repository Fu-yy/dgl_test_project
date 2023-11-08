import os

import new_data
import generate_neg
import argparse


if __name__ == '__main__':
    data_name_list = ["QB-video", "Beauty","cd","test"]
    # data_name_list = ["Games", "Beauty", "Movie", "cd"]
    # 获取当前文件的目录
    current_dir = os.path.dirname(__file__)

    # 拼接文件路径  os.path.dirname(current_dir)获取上一层

    file_path = os.path.join(os.path.dirname(current_dir), 'results')

    # 将斜杠替换为反斜杠
    file_path = file_path.replace('/', '\\')

    if not os.path.exists(file_path):
        os.mkdir(file_path)
    # 测试一个数据集
    new_data.mainLoadData(data_name_list[3], 10, 50, 50, 3, os.path.join(file_path, data_name_list[3]))




    # for i in data_name_list:
    #     new_data.mainLoadData(i, 10, 50, 50, 3, os.path.join(file_path, i))
    #

    generate_neg.gen_neg(data_name_list[3])





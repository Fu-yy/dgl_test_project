import os

import new_data
import argparse


if __name__ == '__main__':
    data_name_list = ["QB-video", "Beauty","cd"]
    # data_name_list = ["Games", "Beauty", "Movie", "cd"]
    # 获取当前文件的目录
    current_dir = os.path.dirname(__file__)

    # 拼接文件路径
    file_path = os.path.join(current_dir, 'results')
    # 将斜杠替换为反斜杠
    file_path = file_path.replace('/', '\\')

    # 测试一个数据集
    new_data.mainLoadData(data_name_list[0], 10, 50, 50, 3, os.path.join(file_path, data_name_list[0]))




    # for i in data_name_list:
    #     new_data.mainLoadData(i, 10, 50, 50, 3, os.path.join(file_path, i))
    #


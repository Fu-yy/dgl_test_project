import pandas as pd


def load_csv_resort_id(data_path):
    """

    :param data_path: 传入csv路径
    :return: 将user_id和item_id从0到n重排序，为了适配建图

    """

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


def print_graph(sub_graph):

    print("print_graph---begin")
    # 输出异构图的节点总数
    num_nodes = sub_graph.num_nodes()
    print("Number of nodes:", num_nodes)

    # 输出异构图的边总数
    num_edges = sub_graph.num_edges()
    print("Number of edges:", num_edges)

    # 输出异构图的节点类型数量
    # num_ntypes = sub_graph.number_of_ntypes()
    # print("Number of node types:", num_ntypes)

    # 输出异构图的边类型数量
    # num_etypes = sub_graph.number_of_etypes()
    # print("Number of edge types:", num_etypes)

    # 输出异构图中的节点类型列表
    node_types = sub_graph.ntypes
    print("Node types:", node_types)

    # 输出异构图中的边类型列表
    edge_types = sub_graph.etypes
    print("Edge types:", edge_types)

    # 输出异构图中所有节点的信息
    print("All nodes:")
    for node_type in node_types:
        nodes = sub_graph.nodes(node_type)
        print(f"Nodes of type {node_type}: {nodes}")
        print(f"{node_type}:节点特征")
        print(sub_graph.nodes[node_type].data)
        # print(sub_graph.nodes[node_type].data[f"{node_type}'_id'"])

    # 输出异构图中所有边的信息
    print("All edges:")
    for edge_type in edge_types:
        edges = sub_graph.edges(form='uv', etype=edge_type)
        print(f"Edges of type {edge_type}: {edges}")

    print(sub_graph.ndata)
    # print(sub_graph.nodes['user'].data)
    # print(sub_graph.nodes['item'].data)
    # print(sub_graph.nodes['user'].data['user_id'])
    # print(sub_graph.nodes['item'].data['item_id'])

    print("print_graph---end")

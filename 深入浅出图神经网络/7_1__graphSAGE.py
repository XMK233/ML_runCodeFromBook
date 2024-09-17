import random, os, tqdm, time, json, re
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from datetime import datetime

import sys

sys.path.append("../../../")

random.seed(618)
np.random.seed(907)

new_base_path = os.path.join(
    "/Users/minkexiu/Downloads/",
    "/".join(
        os.getcwd().split("/")[-1 * (len(sys.path[-1].split("/")) - 1):]
    ),
)
print("storage dir:", new_base_path)
print("code dir:", os.getcwd())

## 创建文件夹。
if not os.path.exists(new_base_path):
    os.makedirs(
        new_base_path
    )
if not os.path.exists(os.path.join(new_base_path, "preprocessedData")):
    os.makedirs(
        os.path.join(new_base_path, "preprocessedData")
    )
if not os.path.exists(os.path.join(new_base_path, "originalData")):
    os.makedirs(
        os.path.join(new_base_path, "originalData")
    )
if not os.path.exists(os.path.join(new_base_path, "trained_models")):
    os.makedirs(
        os.path.join(new_base_path, "trained_models")
    )


def create_originalData_path(filename_or_path):
    return os.path.join(new_base_path, "originalData", filename_or_path)


def create_preprocessedData_path(filename_or_path):
    return os.path.join(new_base_path, "preprocessedData", filename_or_path)


def create_trained_models_path(filename_or_path):
    return os.path.join(new_base_path, "trained_models", filename_or_path)


def millisec2datetime(timestamp):
    time_local = time.localtime(timestamp / 1000)
    return time.strftime("%Y-%m-%d %H:%M:%S", time_local)


def run_finish():
    # 假设你的字体文件是 'myfont.ttf' 并且位于当前目录下
    font = FontProperties(fname="/Users/minkexiu/Documents/GitHub/ML_Tryout/SimHei.ttf", size=24)
    # 创建一个空白的图形
    fig, ax = plt.subplots()
    ax.imshow(
        plt.imread("/Users/minkexiu/Downloads/wallhaven-dgxpyg.jpg")
    )
    # 在图形中添加文字
    ax.text(
        ax.get_xlim()[1] * 0.5,
        ax.get_ylim()[0] * 0.5,
        f"程序于这个点跑完：\n{millisec2datetime(time.time() * 1000)}", fontproperties=font, ha="center", va="center",
        color="red"
    )
    # 设置图形的布局
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.patch.set_color("blue")
    # 显示图形
    plt.show()


tqdm.tqdm.pandas()  ## 引入这个，就可以在apply的时候用progress_apply了。

import IPython


def kill_current_kernel():
    '''杀死当前的kernel释放内存空间。'''
    IPython.Application.instance().kernel.do_shutdown(True)


def simply_show_data(df1):
    print(df1.shape)
    display(df1.head())


def wait_flag(saved_flag_path, time_interval_sec=10):
    print("waiting for", saved_flag_path)
    time_count = 0
    while True:
        if os.path.exists(saved_flag_path):
            break
        time.sleep(time_interval_sec)
        time_count += time_interval_sec
        print(time_count, end=" ")
    print("finish!!")


def parallelly_run_multiple_similar_python_code(codes, nb_workers=4):
    '''
    codes是多条相似的python代码。
    这个函数的作用就是将其平行地跑，每一条python代码就对应一个线程。或许可以后续优化，比如固定线程数为一个特定值。
    nb_workers 如果赋值为
    '''
    assert (isinstance(nb_workers, int)), "`nb_workers' should be int."
    df_sqls = pd.DataFrame(
        {
            "func": codes

        }
    )
    display(df_sqls)
    from pandarallel import pandarallel
    pandarallel.initialize(nb_workers=df_sqls.shape[0] if nb_workers < 0 else nb_workers, progress_bar=True)

    def run_sql_prlly(row):
        try:
            cmd = f'{row["func"]}'
            print(cmd, "\n")
            eval(cmd)
            return "0-success"
        except Exception as e:
            return e

    df_sqls["run_rsts"] = df_sqls.parallel_apply(lambda row: run_sql_prlly(row), axis=1)
    display(df_sqls)


def create_originalData_path(filename_or_path):
    return os.path.join(new_base_path, "originalData", filename_or_path)


def create_preprocessedData_path(filename_or_path):
    return os.path.join(new_base_path, "preprocessedData", filename_or_path)


def create_trained_models_path(filename_or_path):
    return os.path.join(new_base_path, "trained_models", filename_or_path)


class TimerContext:
    def __enter__(self):
        self.start_time = str(datetime.now())
        print("start time:", self.start_time)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("start time:", self.start_time)
        print("end time", str(datetime.now()))


def three_num_get_gua(a, b, c):
    '''梅花易数三数起卦，以取本、互、变。'''
    bagua = ["111", "110", "101", "100", "011", "010", "001", "000"]
    guatu = {
        "111": ("☰", "天", "乾金"),
        "110": ("☱", "泽", "兑金"),
        "101": ("☲", "火", "离火"),
        "100": ("☳", "雷", "震木"),
        "011": ("☴", "风", "巽木"),
        "010": ("☵", "水", "坎水"),
        "001": ("☶", "山", "艮土"),
        "000": ("☷", "地", "坤土"),
    }
    shanggua_idx = 7 if (a % 8 == 0) else (a % 8 - 1)
    xiagua_idx = 7 if (b % 8 == 0) else (b % 8 - 1)
    bianyao_idx = 5 if (c % 6 == 0) else (c % 6 - 1)
    bengua = bagua[xiagua_idx] + bagua[shanggua_idx]
    hugua = bengua[1:-1][:3] + bengua[1:-1][1:]
    biangua = list(bengua)
    biangua[bianyao_idx] = str(1 - int(biangua[bianyao_idx]))
    biangua = "".join(biangua)
    df = pd.DataFrame([[
        guatu[bengua[3:]][0] + guatu[bengua[3:]][2], guatu[hugua[3:]][0] + guatu[hugua[3:]][2],
        guatu[biangua[3:]][0] + guatu[biangua[3:]][2],
    ], [
        guatu[bengua[:3]][0] + guatu[bengua[:3]][2], guatu[hugua[:3]][0] + guatu[hugua[:3]][2],
        guatu[biangua[:3]][0] + guatu[biangua[:3]][2],
    ]], index=["上卦", "下卦"], columns=[
        guatu[bengua[3:]][1] + guatu[bengua[:3]][1],
        guatu[hugua[3:]][1] + guatu[hugua[:3]][1],
        guatu[biangua[3:]][1] + guatu[biangua[:3]][1],
    ])
    display(df)
    return bengua, hugua, biangua


def easy_start_gua():
    """用公历的日、时、分来起卦。"""
    n1, n2, n3 = str(datetime.now())[8:10], str(datetime.now())[11:13], str(datetime.now())[14:16]
    print(n1, n2, n3)
    return three_num_get_gua(int(n1), int(n2), int(n3))


easy_start_gua()

import zhdate


def easy_start_gua_lunar():
    '''用农历的月、日、时辰来起卦。'''
    time_now = datetime.now()
    zh_date_str = str(zhdate.ZhDate.from_datetime(time_now))
    zh_date_str_1 = datetime.strftime(
        datetime(
            *[int(x) for x in re.findall("\d+", zh_date_str)]
        ),
        '%Y-%m-%d'
    )
    zh_hour = (time_now.hour + 1) // 2 % 12 + 1
    zh_hour_dizhi = "子、丑、寅、卯、辰、巳、午、未、申、酉、戌、亥".split("、")[zh_hour - 1]

    n1, n2, n3 = zh_date_str_1[5:7], zh_date_str_1[8:10], zh_hour
    print(n1, n2, n3, f"{zh_hour_dizhi}时")
    return three_num_get_gua(int(n1), int(n2), int(n3))


easy_start_gua_lunar()

import torch

import numpy as np
import torch.nn as nn
import torch.optim as optim
# from net import GraphSage
# from data import CoraData
# from sampling import multihop_sampling

import torch.nn.functional as F
import torch.nn.init as init

from collections import namedtuple

import os
import os.path as osp
import pickle
import itertools
import scipy.sparse as sp
import urllib
from collections import namedtuple


## 定义数据：

class CoraData(object):
    download_url = "https://github.com/kimiyoung/planetoid/raw/master/data"
    filenames = [
        "ind.cora.{}".format(name) for name in
        ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
    ]

    def __init__(self, data_root="cora", rebuild=False):
        """Cora数据，包括数据下载，处理，加载等功能
        当数据的缓存文件存在时，将使用缓存文件，否则将下载、进行处理，并缓存到磁盘

        处理之后的数据可以通过属性 .data 获得，它将返回一个数据对象，包括如下几部分：
            * x: 节点的特征，维度为 2708 * 1433，类型为 np.ndarray
            * y: 节点的标签，总共包括7个类别，类型为 np.ndarray
            * adjacency: 邻接矩阵，维度为 2708 * 2708，类型为 scipy.sparse.coo.coo_matrix
            * train_mask: 训练集掩码向量，维度为 2708，当节点属于训练集时，相应位置为True，否则False
            * val_mask: 验证集掩码向量，维度为 2708，当节点属于验证集时，相应位置为True，否则False
            * test_mask: 测试集掩码向量，维度为 2708，当节点属于测试集时，相应位置为True，否则False

        Args:
        -------
            data_root: string, optional
                存放数据的目录，原始数据路径: ../data/cora
                缓存数据路径: {data_root}/ch5_cached.pkl
            rebuild: boolean, optional
                是否需要重新构建数据集，当设为True时，如果存在缓存数据也会重建数据

        """
        print("init...")
        self.data_root = create_originalData_path(data_root)
        save_file = osp.join(self.data_root, "ch5_cached.pkl")
        if osp.exists(save_file) and not rebuild:
            print("Using Cached file: {}".format(save_file))
            self._data = pickle.load(open(save_file, "rb"))
        else:
            if not osp.exists(self.data_root):
                os.makedirs(self.data_root)
            self._data = self.process_data()
            with open(save_file, "wb") as f:
                pickle.dump(self.data, f)
            print("Cached file: {}".format(save_file))

    @property
    def data(self):
        """返回Data数据对象，包括x, y, adjacency, train_mask, val_mask, test_mask"""
        return self._data

    def process_data(self):
        """
        处理数据，得到节点特征和标签，邻接矩阵，训练集、验证集以及测试集
        引用自：https://github.com/rusty1s/pytorch_geometric
        """
        print("Process data ...")
        _, tx, allx, y, ty, ally, graph, test_index = [
            self.read_data(
                osp.join(self.data_root, name)
            ) for name in self.filenames
        ]
        train_index = np.arange(y.shape[0])
        val_index = np.arange(y.shape[0], y.shape[0] + 500)
        sorted_test_index = sorted(test_index)

        x = np.concatenate((allx, tx), axis=0)
        y = np.concatenate((ally, ty), axis=0).argmax(axis=1)

        x[test_index] = x[sorted_test_index]
        y[test_index] = y[sorted_test_index]
        num_nodes = x.shape[0]

        train_mask = np.zeros(num_nodes, dtype=bool)
        val_mask = np.zeros(num_nodes, dtype=bool)
        test_mask = np.zeros(num_nodes, dtype=bool)
        train_mask[train_index] = True
        val_mask[val_index] = True
        test_mask[test_index] = True
        adjacency = self.build_adjacency(graph)
        print("Node's feature shape: ", x.shape)
        print("Node's label shape: ", y.shape)
        print("Adjacency's shape: ", adjacency.shape)
        print("Number of training nodes: ", train_mask.sum())
        print("Number of validation nodes: ", val_mask.sum())
        print("Number of test nodes: ", test_mask.sum())

        return Data(x=x, y=y, adjacency=adjacency,
                    train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    @staticmethod
    def build_adjacency(adj_dict):
        """根据邻接表创建邻接矩阵"""
        edge_index = []
        num_nodes = len(adj_dict)
        for src, dst in adj_dict.items():
            edge_index.extend([src, v] for v in dst)
            edge_index.extend([v, src] for v in dst)
        # 去除重复的边
        edge_index = list(k for k, _ in itertools.groupby(sorted(edge_index)))
        edge_index = np.asarray(edge_index)
        adjacency = sp.coo_matrix((np.ones(len(edge_index)),
                                   (edge_index[:, 0], edge_index[:, 1])),
                                  shape=(num_nodes, num_nodes), dtype="float32")
        return adjacency

    @staticmethod
    def read_data(path):
        """使用不同的方式读取原始数据以进一步处理"""
        name = osp.basename(path)
        if name == "ind.cora.test.index":
            out = np.genfromtxt(path, dtype="int64")
            return out
        else:
            out = pickle.load(open(path, "rb"), encoding="latin1")
            out = out.toarray() if hasattr(out, "toarray") else out
            return out

    @staticmethod
    def normalization(adjacency):
        """计算 L=D^-0.5 * (A+I) * D^-0.5"""
        adjacency += sp.eye(adjacency.shape[0])  # 增加自连接
        degree = np.array(adjacency.sum(1))
        d_hat = sp.diags(np.power(degree, -0.5).flatten())
        return d_hat.dot(adjacency).dot(d_hat).tocoo()


## 定义网络：

class NeighborAggregator(nn.Module):
    def __init__(self, input_dim, output_dim,
                 use_bias=False, aggr_method="mean"):
        """聚合节点邻居

        Args:
            input_dim: 输入特征的维度
            output_dim: 输出特征的维度
            use_bias: 是否使用偏置 (default: {False})
            aggr_method: 邻居聚合方式 (default: {mean})
        """
        super(NeighborAggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.aggr_method = aggr_method
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, neighbor_feature):
        if self.aggr_method == "mean":
            aggr_neighbor = neighbor_feature.mean(dim=1)
        elif self.aggr_method == "sum":
            aggr_neighbor = neighbor_feature.sum(dim=1)
        elif self.aggr_method == "max":
            aggr_neighbor = neighbor_feature.max(dim=1)
        else:
            raise ValueError("Unknown aggr type, expected sum, max, or mean, but got {}"
                             .format(self.aggr_method))

        neighbor_hidden = torch.matmul(aggr_neighbor, self.weight)
        if self.use_bias:
            neighbor_hidden += self.bias

        return neighbor_hidden

    def extra_repr(self):
        return 'in_features={}, out_features={}, aggr_method={}'.format(
            self.input_dim, self.output_dim, self.aggr_method)


class SageGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 activation=F.relu,
                 aggr_neighbor_method="mean",
                 aggr_hidden_method="sum"):
        """SageGCN层定义

        Args:
            input_dim: 输入特征的维度
            hidden_dim: 隐层特征的维度，
                当aggr_hidden_method=sum, 输出维度为hidden_dim
                当aggr_hidden_method=concat, 输出维度为hidden_dim*2
            activation: 激活函数
            aggr_neighbor_method: 邻居特征聚合方法，["mean", "sum", "max"]
            aggr_hidden_method: 节点特征的更新方法，["sum", "concat"]
        """
        super(SageGCN, self).__init__()
        assert aggr_neighbor_method in ["mean", "sum", "max"]
        assert aggr_hidden_method in ["sum", "concat"]
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.aggr_neighbor_method = aggr_neighbor_method
        self.aggr_hidden_method = aggr_hidden_method
        self.activation = activation
        self.aggregator = NeighborAggregator(input_dim, hidden_dim,
                                             aggr_method=aggr_neighbor_method)
        self.weight = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)

    def forward(self, src_node_features, neighbor_node_features):
        neighbor_hidden = self.aggregator(neighbor_node_features)
        self_hidden = torch.matmul(src_node_features, self.weight)

        if self.aggr_hidden_method == "sum":
            hidden = self_hidden + neighbor_hidden
        elif self.aggr_hidden_method == "concat":
            hidden = torch.cat([self_hidden, neighbor_hidden], dim=1)
        else:
            raise ValueError("Expected sum or concat, got {}"
                             .format(self.aggr_hidden))
        if self.activation:
            return self.activation(hidden)
        else:
            return hidden

    def extra_repr(self):
        output_dim = self.hidden_dim if self.aggr_hidden_method == "sum" else self.hidden_dim * 2
        return 'in_features={}, out_features={}, aggr_hidden_method={}'.format(
            self.input_dim, output_dim, self.aggr_hidden_method)


class GraphSage(nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 num_neighbors_list):
        super(GraphSage, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_neighbors_list = num_neighbors_list
        self.num_layers = len(num_neighbors_list)
        self.gcn = nn.ModuleList()
        self.gcn.append(SageGCN(input_dim, hidden_dim[0]))
        for index in range(0, len(hidden_dim) - 2):
            self.gcn.append(SageGCN(hidden_dim[index], hidden_dim[index + 1]))
        self.gcn.append(SageGCN(hidden_dim[-2], hidden_dim[-1], activation=None))

    def forward(self, node_features_list):
        hidden = node_features_list
        for l in range(self.num_layers):
            next_hidden = []
            gcn = self.gcn[l]
            for hop in range(self.num_layers - l):
                src_node_features = hidden[hop]
                src_node_num = len(src_node_features)
                neighbor_node_features = hidden[hop + 1] \
                    .view((src_node_num, self.num_neighbors_list[hop], -1))
                h = gcn(src_node_features, neighbor_node_features)
                next_hidden.append(h)
            hidden = next_hidden
        return hidden[0]

    def extra_repr(self):
        return 'in_features={}, num_neighbors_list={}'.format(
            self.input_dim, self.num_neighbors_list
        )


## 定义采样的方法：

def sampling(src_nodes, sample_num, neighbor_table):
    """根据源节点采样指定数量的邻居节点，注意使用的是有放回的采样；
    某个节点的邻居节点数量少于采样数量时，采样结果出现重复的节点

    Arguments:
        src_nodes {list, ndarray} -- 源节点列表
        sample_num {int} -- 需要采样的节点数
        neighbor_table {dict} -- 节点到其邻居节点的映射表

    Returns:
        np.ndarray -- 采样结果构成的列表
    """
    # print("neighbor_table type:", type(neighbor_table))
    results = []
    for sid in src_nodes:
        # 从节点的邻居中进行有放回地进行采样
        res = np.random.choice(
            np.where(data.adjacency_dict.todense()[113] == 1)[1],
            size=(sample_num,)
        )  ## np.random.choice(neighbor_table.tocsr()[sid], size=(sample_num, ))
        ## xmk；这里跟原来的代码有点不一样。原来的代码有问题，查了一下是因为coo类型的矩阵没法是没法index的。
        ## 然后观察了一下，这里代码要实现的意思。所以就自己找了个方法给它修改了，就能跑了。
        results.append(res)
    return np.asarray(results).flatten()


def multihop_sampling(src_nodes, sample_nums, neighbor_table):
    """根据源节点进行多阶采样

    Arguments:
        src_nodes {list, np.ndarray} -- 源节点id
        sample_nums {list of int} -- 每一阶需要采样的个数
        neighbor_table {dict} -- 节点到其邻居节点的映射

    Returns:
        [list of ndarray] -- 每一阶采样的结果
    """
    sampling_result = [src_nodes]
    for k, hopk_num in enumerate(sample_nums):
        # print(sampling_result[k], type(sampling_result[k]))
        hopk_result = sampling(sampling_result[k], hopk_num, neighbor_table)
        sampling_result.append(hopk_result)
    return sampling_result

INPUT_DIM = 1433    # 输入维度
# Note: 采样的邻居阶数需要与GCN的层数保持一致
HIDDEN_DIM = [128, 7]   # 隐藏单元节点数
NUM_NEIGHBORS_LIST = [10, 10]   # 每阶采样邻居的节点数
assert len(HIDDEN_DIM) == len(NUM_NEIGHBORS_LIST)
BTACH_SIZE = 16     # 批处理大小
EPOCHS = 20
NUM_BATCH_PER_EPOCH = 20    # 每个epoch循环的批次数
LEARNING_RATE = 0.01    # 学习率
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

Data = namedtuple('Data', ['x', 'y', 'adjacency_dict',
                           'train_mask', 'val_mask', 'test_mask'])

data = CoraData().data
x = data.x / data.x.sum(1, keepdims=True)  # 归一化数据，使得每一行和为1

train_index = np.where(data.train_mask)[0]
train_label = data.y
test_index = np.where(data.test_mask)[0]
model = GraphSage(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,
                  num_neighbors_list=NUM_NEIGHBORS_LIST).to(DEVICE)
print(model)
criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)

def train():
    model.train()
    for e in range(EPOCHS):
        for batch in range(NUM_BATCH_PER_EPOCH):
            batch_src_index = np.random.choice(train_index, size=(BTACH_SIZE,))
            batch_src_label = torch.from_numpy(train_label[batch_src_index]).long().to(DEVICE)
            batch_sampling_result = multihop_sampling(batch_src_index, NUM_NEIGHBORS_LIST, data.adjacency_dict)
            batch_sampling_x = [torch.from_numpy(x[idx]).float().to(DEVICE) for idx in batch_sampling_result]
            batch_train_logits = model(batch_sampling_x)
            loss = criterion(batch_train_logits, batch_src_label)
            optimizer.zero_grad()
            loss.backward()  # 反向传播计算参数的梯度
            optimizer.step()  # 使用优化方法进行梯度更新
            print("Epoch {:03d} Batch {:03d} Loss: {:.4f}".format(e, batch, loss.item()))
        test()


def test():
    model.eval()
    with torch.no_grad():
        test_sampling_result = multihop_sampling(test_index, NUM_NEIGHBORS_LIST, data.adjacency_dict)
        test_x = [torch.from_numpy(x[idx]).float().to(DEVICE) for idx in test_sampling_result]
        test_logits = model(test_x)
        test_label = torch.from_numpy(data.y[test_index]).long().to(DEVICE)
        predict_y = test_logits.max(1)[1]
        accuarcy = torch.eq(predict_y, test_label).float().mean().item()
        print("Test Accuracy: ", accuarcy)

train()
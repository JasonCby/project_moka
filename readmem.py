import os
import os.path as osp
import time
import os, mmap
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import add_self_loops
from sklearn.manifold import TSNE
import numpy as np

path = "/dev/pmem0"
cites = path + "cora.cites"
content = path + "cora.content"

# 索引字典，将原本的论文id转换到从0开始编码
index_dict = dict()
# 标签字典，将字符串标签转化为数值
label_to_index = dict()

features = []
labels = []
edge_index = []
# start timer
start_time = time.perf_counter()

with mmap.mmap(os.open(path, os.O_RDONLY), 0) as f:
    nodes = []
    while True:
        text_line = f.readline().decode().strip()
        if text_line:
            nodes.append(text_line)
        else:
            break
    print(nodes[0])
    print(len(nodes))
    #nodes = f.readlines()
    for node in nodes:
        node_info = node.split()
        #print(node_info[0])
        index_dict[int(node_info[0])] = len(index_dict)
        features.append([int(i) for i in node_info[1:-1]])

        label_str = node_info[-1]
        if (label_str not in label_to_index.keys()):
            label_to_index[label_str] = len(label_to_index)
        labels.append(label_to_index[label_str])
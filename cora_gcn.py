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

# import matplotlib.pyplot as plt

# from pmem
#path = "/mnt/mem/python/project_moka/data/Cora/"
# form ram
path = "data/Cora/"
#path = "/dev/pmem0"
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
fd = os.open(content, os.O_RDWR)

with mmap.mmap(fd, 0) as f:
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

with mmap.mmap(os.open(cites, os.O_RDWR), 0) as f:
    edges = []
    while True:
        text_line = f.readline().decode().strip()
        if text_line:
            edges.append(text_line)
        else:
            break
    for edge in edges:
        start, end = edge.split()
        # 训练时将边视为无向的，但原本的边是有向的，因此需要正反添加两次
        edge_index.append([index_dict[int(start)], index_dict[int(end)]])
        edge_index.append([index_dict[int(end)], index_dict[int(start)]])

# 为每个节点增加自环，但后续GCN层默认会添加自环，跳过即可
# for i in range(2708):
#     edge_index.append([i,i])

# 转换为Tensor
labels = torch.LongTensor(labels)
features = torch.FloatTensor(features)
# 行归一化
# features = torch.nn.functional.normalize(features, p=1, dim=1)
edge_index = torch.LongTensor(edge_index)


class GCNNet(torch.nn.Module):
    def __init__(self, num_feature, num_label):
        super(GCNNet, self).__init__()
        self.GCN1 = GCNConv(num_feature, 16)
        self.GCN2 = GCNConv(16, num_label)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.GCN1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.GCN2(x, edge_index)

        return F.log_softmax(x, dim=1)


class GATNet(torch.nn.Module):
    def __init__(self, num_feature, num_label):
        super(GATNet, self).__init__()
        self.GAT1 = GATConv(num_feature, 8, heads=8, concat=True, dropout=0.6)
        self.GAT2 = GATConv(8 * 8, num_label, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.GAT1(x, edge_index)
        x = F.relu(x)
        x = self.GAT2(x, edge_index)

        return F.log_softmax(x, dim=1)


seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)  # Numpy module.
# random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

mask = torch.randperm(len(index_dict))
train_mask = mask[:140]
val_mask = mask[140:640]
test_mask = mask[1708:2708]

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

cora = Data(x=features, edge_index=edge_index.t().contiguous(), y=labels).to(device)
#model
model = GATNet(features.shape[1], len(label_to_index)).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
mean_time = 0
total_time = 0
times = 1

for _ in range(times):

    mid = time.perf_counter()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(cora)
        loss = F.nll_loss(out[train_mask], cora.y[train_mask])
        #print('epoch: %d loss: %.4f' % (epoch, loss))
        loss.backward()
        optimizer.step()

        if ((epoch + 1) % 10 == 0):
            model.eval()
            _, pred = model(cora).max(dim=1)
            correct = int(pred[test_mask].eq(cora.y[test_mask]).sum().item())
            acc = correct / len(test_mask)
            #print('Accuracy: {:.4f}'.format(acc))
            model.train()
    # stop timer
    end = time.perf_counter()
    # output duration
    duration = end - start_time
    print('Running time: %s Seconds' % duration)
    print('Running 2nd time: %s Seconds' % (end - mid))
    print('Difference: %s Seconds' % (mid - start_time))
    total_time += duration
mean_time = total_time/times
print('Mean running time: %s Seconds' % mean_time)

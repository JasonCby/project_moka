import time

import torch
from torch_geometric.datasets import Planetoid, WikiCS, ShapeNet, Reddit, Reddit2, CoMA, AmazonProducts
from torch_geometric.transforms import NormalizeFeatures
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

path = "/mnt/NVme/project_moka/pubmed/"
path = "/mnt/ramfs/project_moka/pubmed/"
#path = "/mnt/ext4ramdisk/project_moka/pubmed/"
#path = "./pubmed/"

path_Cora = "/mnt/NVme/project_moka/data/Cora/"
# path_Cora = "/mnt/ramfs/project_moka/data/Cora/"
#path_Cora = "/mnt/ext4ramdisk/project_moka/data/Cora/"
#path_Cora = "./data/Cora/"

path_pm = "/mnt/NVme/datasets/"
path_ram = "/mnt/ramfs/datasets/"

times = 15
total_time = 0
total_run_time = 0
batch_size = 128
epoch_num = 20

# pre-load Planetoid
dataset_test = Planetoid(root="./pubmed/", name='Pubmed')
#dataset_test = Planetoid(root=path_Cora, name='Cora')
# the dataset for test is shown below (different from the above)
# dataset_test = Planetoid(root='./data/Cora/', name='Cora')


dataset = dataset_test


data = dataset[0]  # Get the first graph object.

from torch_geometric.data import ClusterData, ClusterLoader, DataLoader

torch.manual_seed(32322)
cluster_data = ClusterData(data, num_parts=128)  # 1. Create subgraphs.
train_loader = ClusterLoader(cluster_data, batch_size=batch_size,
                             shuffle=True)  # 2. Stochastic partitioning scheme.


# finish pre-load

class GATNet(torch.nn.Module):
    def __init__(self):
        super(GATNet, self).__init__()
        self.gat1 = GATConv(dataset.num_node_features, 8, 8, dropout=0.6)
        self.gat2 = GATConv(64, 7, 1, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gat1(x, edge_index)
        x = self.gat2(x, edge_index)
        return F.log_softmax(x, dim=1)


class SAGEConvNet(torch.nn.Module):
    def __init__(self):
        super(SAGEConvNet, self).__init__()
        self.conv1 = SAGEConv(dataset.num_node_features, 16)
        self.conv2 = SAGEConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.softmax(x, dim=1)

        return x


class GCNNet(torch.nn.Module):
    def __init__(self):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


for n in range(1):

    # start timer
    start = time.perf_counter()
    dataset_pubmed1 = AmazonProducts(root="/mnt/NVMe/project_moka/datasets/")
    #dataset_pubmed = Planetoid(root="./pubmed/")
    #dataset_Cora = Planetoid(root=path_Cora, name='Cora', split="random")
    # start timer
    after = time.perf_counter()

    dataset_pubmed2 = AmazonProducts(root="/mnt/NVMe/project_moka/datasets/")
    #dataset_Cora = Planetoid(root=path_Cora, name='Cora', split="random")
    # start timer
    after2 = time.perf_counter()

    #dataset = dataset_pubmed
    dataset = dataset_pubmed2
    data = dataset[0]  # Get the first graph object.

    # from torch_geometric.data import ClusterData, ClusterLoader, DataLoader
    #
    # torch.manual_seed(32322)
    # cluster_data = ClusterData(data, num_parts=128)  # 1. Create subgraphs.
    # train_loader = ClusterLoader(cluster_data, batch_size=batch_size,
    #                              shuffle=True)  # 2. Stochastic partitioning scheme.
    # mid = time.perf_counter()
    # print()
    # total_num_nodes = 0
    # for step, sub_data in enumerate(train_loader):
    #     total_num_nodes += sub_data.num_nodes

    # print(f'Iterated over {total_num_nodes} of {data.num_nodes} nodes!')

    # from IPython.display import Javascript, display

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    #
    model = GCNNet().to(device)
    #
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()

    train_start = time.perf_counter()
    ###
    # for epoch in range(epoch_num):
    #     batch_round = 0
    #     for train_data in train_loader:
    #         batch_round += 1
    #         data = train_data.to(device)
    #         optimizer.zero_grad()
    #         out = model(data)
    #         loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    #         loss.backward()
    #         optimizer.step()
    ###
    end = time.perf_counter()

    # output duration
    #loader_time = mid - after
    train_time = end - train_start
    file_reading = after - start
    file_reading2 = after2 - after
    print('Reading time: %s Seconds' % file_reading)
    print('Reading time2: %s Seconds' % file_reading2)
    #print('Loader time: %s Seconds' % loader_time)
    print('Training time: %s Seconds' % train_time)
    #model.eval()
    #_, pred = model(data).max(dim=1)
    #correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    #acc = correct / int(data.test_mask.sum())
    # print('Accuracy:{:.4f}'.format(acc))
    if n != 0:
        total_time += file_reading

mean_time = total_time / (times - 1)
print('Mean reading time: %s Seconds' % mean_time)
# print('Mean training time: %s Seconds' % mean_run_time)

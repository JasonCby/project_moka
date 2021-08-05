import time

import torch
from torch_geometric.datasets import Planetoid, WikiCS, ShapeNet, Reddit, Reddit2, CoMA, AmazonProducts
from torch_geometric.transforms import NormalizeFeatures

path = "/mnt/mem/project_moka/pubmed/"
#path = "/mnt/tmpfs/project_moka/pubmed/"
#path = "/mnt/ext4ramdisk/project_moka/pubmed/"
#path = "./pubmed/"
times = 10
total_time = 0
batch_size = 32
epoch_num = 10

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


for _ in range(times):
    # start timer
    start = time.perf_counter()


    dataset_pubmed = Planetoid(root=path, name='Pubmed')

    # dataset_Reddit = Reddit(root='./reddit/')
    # dataset_Reddit2 = Reddit2(root='./reddit2/')
    # dataset_AmazonProducts = AmazonProducts(root='./AmazonProducts')
    # dataset_wiki = WikiCS(root='./WikiCS/')
    dataset = dataset_pubmed
    data = dataset[0]  # Get the first graph object.
    ''' print(f'Dataset: {dataset}:')
    print('==================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    

    print(data)
    print('===============================================================================================================')

    # Gather some statistics about the graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Number of training nodes: {data.train_mask.sum()}')
    print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.3f}')
    print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
    print(f'Contains self-loops: {data.contains_self_loops()}')'''
    # print(f'Is undirected: {data.is_undirected()}')

    from torch_geometric.data import ClusterData, ClusterLoader, DataLoader

    torch.manual_seed(23122)
    cluster_data = ClusterData(data, num_parts=128)  # 1. Create subgraphs.
    train_loader = ClusterLoader(cluster_data, batch_size=batch_size, shuffle=True)  # 2. Stochastic partioning scheme.

    print()
    total_num_nodes = 0
    for step, sub_data in enumerate(train_loader):
        #print(f'Step {step + 1}:')
        #print('=======')
        #print(f'Number of nodes in the current batch: {sub_data.num_nodes}')
        #print(sub_data)
        #print()
        total_num_nodes += sub_data.num_nodes

    #print(f'Iterated over {total_num_nodes} of {data.num_nodes} nodes!')

    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, SAGEConv, GATConv

    #from IPython.display import Javascript, display

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    #
    model = GCNNet().to(device)
    #
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()


    for epoch in range(epoch_num):
        #print("Epoch:" + str(epoch))
        batch_round = 0
        for train_data in train_loader:
            batch_round += 1
            data = train_data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            #print("Epoch:" + str(epoch) + ". Batch: " + str(batch_round) + ".")
        #print("Epoch " + str(epoch_num) + " Done!")
        # stop timer
    end = time.perf_counter()
    # output duration
    duration = end - start
    print('Running time: %s Seconds' % duration)

    model.eval()
    _, pred = model(data).max(dim=1)
    correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / int(data.test_mask.sum())
    #print('Accuracy:{:.4f}'.format(acc))
    total_time += duration

mean_time = total_time / times
#print('Mean running time: %s Seconds' % mean_time)

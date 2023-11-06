import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

# 定义GCN模型
class GCNModelWithWeights(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNModelWithWeights, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data, adj_matrix):
        x, edge_index = data.x, data.edge_index
        
        
        # 使用邻接矩阵的权重来调整邻接节点的特征
        # adj_weighted = torch.mm(adj_matrix, x)

        # 使用协方差矩阵作为边权重
        # 取上三角部分（不包括对角线）
        upper_triangular = torch.triu(adj_matrix, diagonal=1)
        # 展开为一维张量
        adj_flat = upper_triangular[upper_triangular != 0]
        edge_weight = adj_flat

        # adj_weighted = torch.mm(torch.mm(adj_matrix, torch.eye(adj_matrix.shape[0])), x)
        
        # x = self.conv1(adj_weighted, edge_index, edge_weight=edge_weight)
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        # x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        # x = self.dropout(x)

        # 对每一列进行 softmax
        x = F.softmax(x, dim=0)
        x = x.T    # 对解进行转置，得到(100, 31)的解
        return x


class GNN(object):
    def __init__(self, dim, lr, epoches, r, s, c):
        self.dim = dim 
        self.lr = lr 
        self.epoches = epoches 
    
        self.model = GCNModelWithWeights(input_dim=2, hidden_dim=self.dim, output_dim=100)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.r = torch.from_numpy(r).squeeze()   # 将r从 numpy (31,1)转为torch.size([31])
        self.s = torch.tensor(s).squeeze()
        self.c = torch.tensor(c)                 # 将c从 numpy (31,31)转为torch.size([31, 31])


    # 初始化损失函数， 目标函数
    def loss_fun(self, P):

        # 对于objective 1: -return
        M = -torch.matmul(P.double(), self.r)  # 结果形状为 [100]   # 这里的return其实是-return,也就是M越小越好
        # 创建一个与 M 形状相同的张量，所有元素都为1
        target1 = -torch.ones_like(M)
        # 计算 MSE 损失
        return_loss = F.mse_loss(M, target1)


        # 对于objective 2: risk
        temp = P.double() * self.s  # Element-wise乘法
        V = torch.sum(temp.unsqueeze(2) * temp.unsqueeze(1) * self.c, dim=(1, 2))  # 先通过unsqueeze扩展张量的维度，然后进行element-wise乘法和求和操作
        target2 = torch.zeros_like(V)
        risk_loss = F.mse_loss(V, target2)


        # 将M和V组合为一个新的张量
        objs = torch.stack([M, V], dim=1)
        # # 对每一列求和
        # return_sum = torch.sum(objs[:, 0])
        # risk_sum = torch.sum(objs[:, 1])

        # print(return_sum.shape)
        total_loss = return_loss + risk_loss
        # print(objs)
        # print(f'return_loss:{return_loss}\t risk_loss: {risk_loss}')
        return total_loss

    # 创建图数据
    def create_graph_data(self, means, std_devs, cov_matrix):
        num_nodes = len(means)
        
        # 创建节点特征（均值和标准差）
        node_features = torch.tensor(list(zip(means, std_devs)), dtype=torch.float32)
        
        # 创建邻接矩阵（协方差矩阵）
        adj_matrix = np.array(cov_matrix, dtype=np.float32)
        np.fill_diagonal(adj_matrix, 0.0)
        adj_matrix = torch.FloatTensor(adj_matrix)
        # adj_matrix /= torch.max(adj_matrix)
        
        # 创建图数据
        # edge_index = torch.tensor(np.triu_indices(num_nodes, 1), dtype=torch.long)
        # 修改后的代码
        edge_index_np = np.array(np.triu_indices(num_nodes, 1))  # 无向图，只存储一个方向的边即可
        edge_index = torch.tensor(edge_index_np, dtype=torch.long)   # 单向的边

        data = Data(x=node_features, edge_index=edge_index)
        
        return data, adj_matrix

    def train(self):
        
        self.model.train()
        data, adj_matrix = self.create_graph_data(self.r, self.s, self.c)

        for epoch in range(self.epoches):

            self.optimizer.zero_grad()
            # 前向传播
            output = self.model(data, adj_matrix)  # output.shape=(100,31)
            loss = self.loss_fun(output) * 1e2
            # break

            # print("loss: ", loss)
            # print(sum(output[0]))  # output.T.shape=(100, 31)
            # break
            
            # 计算损失（最小化风险）
            # target_risk = torch.std(output)
            # print(target_risk)
            # break
            # loss = criterion(target_risk, torch.zeros(1))
            
            # 反向传播和优化
            loss.backward()
            self.optimizer.step()
            
            # print(f'Epoch [{epoch+1}/{self.epoches}], Loss: {loss.item()}')
            # if (epoch + 1) % 100 == 0:
            #     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

    def generate(self):

        data, adj_matrix = self.create_graph_data(self.r, self.s, self.c)
        # 测试模型
        self.model.eval()
        with torch.no_grad():
            predicted = self.model(data, adj_matrix).cpu().data.numpy()

        return predicted


# 初始化模型
# means: 收益的均值
# std_devs: 收益的标准差
# cov_matrix：不同股份收益的协方差矩阵

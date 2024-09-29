import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
from torch.nn import BatchNorm1d

# 定义 GIN 模型的一个层，包含原子图的卷积
class GINLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GINLayer, self).__init__()
        # GIN 层中的多层感知机（MLP）
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        # GIN 卷积层
        self.conv = GINConv(self.mlp)
        self.batch_norm = BatchNorm1d(out_channels)

    def forward(self, x, edge_index):
        # GIN 卷积层处理
        x = self.conv(x, edge_index)
        x = self.batch_norm(x)
        x = F.relu(x)
        return x

# 定义 GIN 模型，包含多个 GINLayer
class GIN(nn.Module):
    def __init__(self, node_in_dim, node_hidden_dim, num_layers, output_dim):
        super(GIN, self).__init__()
        # 初始化多个 GIN 层
        self.layers = nn.ModuleList()
        self.layers.append(GINLayer(node_in_dim, node_hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GINLayer(node_hidden_dim, node_hidden_dim))

        # 图全局池化
        self.pool = global_mean_pool

        # 最后的全连接层，用于预测吸附量
        self.fc = nn.Linear(node_hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 遍历每一层 GINLayer
        for layer in self.layers:
            x = layer(x, edge_index)

        # 对图进行全局池化（将所有节点的特征聚合为图的一个嵌入）
        x = self.pool(x, batch)

        # 最后的全连接层
        x = self.fc(x)
        return x

# 定义整个模型，包括温度和压力部分的处理
class FullModel(nn.Module):
    def __init__(self, gin_node_in_dim, gin_node_hidden_dim, gin_num_layers, gin_output_dim, temp_pressure_dim):
        super(FullModel, self).__init__()
        # GIN 模型处理图数据部分
        self.gin = GIN(gin_node_in_dim, gin_node_hidden_dim, gin_num_layers, gin_output_dim)

        # 处理温度和压力输入的全连接层
        self.fc_temp_pressure = nn.Linear(temp_pressure_dim, gin_output_dim)

        # 使用 DNN 替代全连接层
        self.dnn = nn.Sequential(
            nn.Linear(2 * gin_output_dim, 64),  # GIN 和温度压力输出合并
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 最后回归输出 1 个值
        )


    def forward(self, data, temp_pressure):
            # 图数据通过 GIN 模型
            graph_output = self.gin(data)

            # 温度和压力的全连接处理
            temp_pressure_output = F.relu(self.fc_temp_pressure(temp_pressure))

            # 合并 GIN 和温度压力输出
            combined = torch.cat([graph_output, temp_pressure_output], dim=1)

            # 最后预测吸附量
            output = self.dnn(combined)

            return output.squeeze()

# 示例用法
if __name__ == "__main__":
    # 模拟数据: 节点输入维度 7，隐藏层 64，3 层，输出 64 维，温度和压力输入维度为 2
    model = FullModel(gin_node_in_dim=7, gin_node_hidden_dim=64, gin_num_layers=3, gin_output_dim=64, temp_pressure_dim=2)

    # 假设我们有图数据（data）和温度压力数据（temp_pressure）
    # data 是从图数据集中加载的 PyTorch Geometric Data 对象，包含 x 和 edge_index
    data = ...  # 从你的数据集中获取
    temp_pressure = torch.tensor([[300.0, 0.1], [310.0, 0.15]], dtype=torch.float)  # 示例输入

    # 前向传播
    prediction = model(data, temp_pressure)
    print("Prediction:", prediction)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import BatchNorm1d

def aggregate_bond_to_atom(bond_to_atom_message, edge_index, num_atoms):
    row, col = edge_index
    atom_message = torch.zeros((num_atoms, bond_to_atom_message.size(1)), device=bond_to_atom_message.device)

    # 聚合到 row 上
    for i in range(len(row)):
        atom_message[row[i]] += bond_to_atom_message[i]
    # 聚合到 col 上
    for i in range(len(col)):
        atom_message[col[i]] += bond_to_atom_message[i]

    return atom_message

# 定义ALIGNN中的一个单层，包含原子图和线图的信息交互
class ALIGNNLayer(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ALIGNNLayer, self).__init__()
        # 原子图上的卷积
        self.conv_atom = GCNConv(in_channels, out_channels)
        # 添加批归一化
        self.batch_norm = BatchNorm1d(out_channels)

        # 线图上的卷积
        self.conv_bond = GCNConv(out_channels, out_channels)

        # 融合来自原子图和线图的信息
        self.atom_to_bond = nn.Linear(out_channels, out_channels)
        self.bond_to_atom = nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index, line_graph_edge_index):
        # print(f"x size: {x.size(0)}")
        # print(f"edge_index max: {edge_index.max()}, edge_index size: {edge_index.size()}")
        # print(
        #     f"line_graph_edge_index max: {line_graph_edge_index.max()}, line_graph_edge_index size: {line_graph_edge_index.size()}")
        # max_line_graph_idx = line_graph_edge_index.max().item()
        # num_edges = edge_index.size(1)  # 原图的边数
        #
        # if max_line_graph_idx >= num_edges:
        #     raise ValueError(
        #         f"line_graph_edge_index contains out-of-bound index {max_line_graph_idx}, num_edges: {num_edges}")

        # 在原子图上进行卷积
        x_atom = self.conv_atom(x, edge_index)
        # x_atom = self.batch_norm(x_atom)
        x_atom = F.relu(x_atom)

        # 打印特征和索引的大小
        # print(f"x_atom size: {x_atom.size()}, line_graph_edge_index size: {line_graph_edge_index.size()}")

        # 从原子特征生成边特征 (可以选择拼接或平均)
        # 例如，通过拼接边的两个原子的特征来生成边的特征
        row, col = edge_index  # 获取原图中的边索引
        x_bond = (x_atom[row] + x_atom[col]) / 2 #x_bond = torch.cat([x_atom[row], x_atom[col]], dim=-1)  # 将边的两个原子的特征拼接起来

        # 将边特征输入到线图卷积中
        x_bond = self.conv_bond(x_bond, line_graph_edge_index)
        # x_bond = self.batch_norm(x_bond)
        x_bond = F.relu(x_bond)

        # 从线图更新到原子图（信息交互）
        bond_to_atom_message = self.bond_to_atom(x_bond)

        # 聚合边特征到原子（基于 row 和 col）
        atom_message = torch.zeros_like(x_atom)
        atom_message = aggregate_bond_to_atom(bond_to_atom_message, edge_index, x_atom.size(0))

        # 将聚合后的边信息与原子特征相加
        x_atom = x_atom + atom_message

        # 从原子图更新到线图（信息交互）
        atom_to_bond_message = self.atom_to_bond(x_atom)
        atom_to_bond_message = F.relu(atom_to_bond_message)

        # 将原子特征映射到边特征，并与 x_bond 相加
        atom_to_bond_message = (atom_to_bond_message[row] + atom_to_bond_message[col]) / 2
        x_bond = x_bond + atom_to_bond_message  # 融合原子图信息到线图

        return x_atom, x_bond


    # 定义ALIGNN模型，包含多个ALIGNNLayer
class ALIGNN(nn.Module):
    def __init__(self, node_in_dim, node_hidden_dim, num_layers, output_dim):
        super(ALIGNN, self).__init__()
        # 初始化多个ALIGNN层
        self.layers = nn.ModuleList()
        self.layers.append(ALIGNNLayer(node_in_dim, node_hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(ALIGNNLayer(node_hidden_dim, node_hidden_dim))

        # 图全局池化
        self.pool = global_mean_pool

        # 最后的全连接层，用于预测吸附量
        self.fc = nn.Linear(node_hidden_dim, output_dim)


    def forward(self, data):
        x, edge_index, line_graph_edge_index = data.x, data.edge_index, data.line_graph_edge_index

        # 遍历每一层ALIGNN层
        for layer in self.layers:
            x, _ = layer(x, edge_index, line_graph_edge_index)

        # 对图进行全局池化（将所有节点的特征聚合为图的一个嵌入）
        x = self.pool(x, data.batch)

        # 最后的全连接层
        x = self.fc(x)
        return x

# 定义整个模型，包括温度和压力部分的处理
class FullModel(nn.Module):
    def __init__(self, alignn_node_in_dim, alignn_node_hidden_dim, alignn_num_layers, alignn_output_dim, temp_pressure_dim):
        super(FullModel, self).__init__()
        # ALIGNN 模型处理图数据部分
        self.alignn = ALIGNN(alignn_node_in_dim, alignn_node_hidden_dim, alignn_num_layers, alignn_output_dim)

        # 处理温度和压力输入的全连接层
        self.fc_temp_pressure = nn.Linear(temp_pressure_dim, alignn_output_dim)
        #self.batch_norm = BatchNorm1d(16)

        # 最后预测吸附量的全连接层
        self.fc1 = nn.Linear(2 * alignn_output_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        #self.fc2 = nn.Linear(16, 32)
        #self.fc3 = nn.Linear(32, 16)
        self.final = nn.Linear(16, 1)

    def forward(self, data, temp_pressure):
        # 图数据通过 ALIGNN 模型
        graph_output = self.alignn(data)

        # 温度和压力的全连接处理
        temp_pressure_output = F.relu(self.fc_temp_pressure(temp_pressure))

        # 合并 ALIGNN 和温度压力输出
        combined = torch.cat([graph_output, temp_pressure_output], dim=1)
        # Batch Normalization
        # normed_combined = self.batch_norm(combined)
        # 最后预测吸附量
        output = self.fc1(combined)
        #output = self.fc1(normed_combined)
        output = F.relu(output)
        output = self.fc2(output)
        output = F.relu(output)
        output = self.final(output)
        # output = F.relu(output)
        # print('output',output)
        return output.squeeze()

# 示例用法
if __name__ == "__main__":
    # 模拟数据: 节点输入维度 7，隐藏层 64，3 层，输出 64 维，温度和压力输入维度为 2
    model = FullModel(alignn_node_in_dim=7, alignn_node_hidden_dim=64, alignn_num_layers=3, alignn_output_dim=64, temp_pressure_dim=2)

    # 假设我们有图数据（data）和温度压力数据（temp_pressure）
    # data 是从图数据集中加载的 PyTorch Geometric Data 对象，包含 x, edge_index 和 line_graph_edge_index
    # temp_pressure 是一个包含温度和压力的 2 维张量
    data = ...  # 从你的数据集中获取
    temp_pressure = torch.tensor([[300.0, 0.1], [310.0, 0.15]], dtype=torch.float)  # 示例输入

    # 前向传播
    prediction = model(data, temp_pressure)
    print("Prediction:", prediction)

import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from ALIDNN import FullModel
from dataset import AdsorptionDataset
import pandas as pd

# 定义训练函数
def train(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch_data in data_loader:
        # 将图数据部分和 temp_pressure 分开处理
        batch_data = batch_data.to(device)
        num_nodes = batch_data.x.size(0)
        max_edge_index = batch_data.edge_index.max().item()
        max_line_graph_index = batch_data.line_graph_edge_index.max().item()



        if max_edge_index >= num_nodes or max_line_graph_index >= batch_data.edge_index.size(1):
            print(
                f"num_nodes: {num_nodes}, max_edge_index: {max_edge_index}, max_line_graph_index: {max_line_graph_index}")
            print("Error: Invalid edge index or line graph edge index!")
            continue

        # 获取图批次信息
        batch_size = batch_data.y.size(0)

        # 确保 temp_pressure 与每个图样本一一对应
        temp_pressure = batch_data.temp_pressure.view(batch_size, -1).to(device)

        labels = batch_data.y.to(device)

        optimizer.zero_grad()

        # 模型前向传播
        predictions = model(batch_data, temp_pressure)
        # print('pre',predictions)
        # print('y',labels)

        # 计算损失
        loss = criterion(predictions, labels)
        loss.backward()

        # 更新模型参数
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


# 定义验证函数
def validate(model, data_loader, criterion, device, mean_y, std_y):
    model.eval()
    total_loss = 0
    all_pre = []
    all_y = []
    with torch.no_grad():
        for batch_data in data_loader:
            # 将图数据部分和 temp_pressure 分开处理
            batch_data = batch_data.to(device)

            # 获取图批次信息
            batch_size = batch_data.y.size(0)

            # 确保 temp_pressure 与每个图样本一一对应
            temp_pressure = batch_data.temp_pressure.view(batch_size, -1).to(device)

            labels = batch_data.y.to(device)


            # 模型前向传播
            predictions = model(batch_data, temp_pressure)

            # 计算损失
            loss = criterion(predictions, labels)
            total_loss += loss.item()
            predictions = predictions * std_y + mean_y
            labels = labels * std_y + mean_y
            for i in predictions.detach().cpu().numpy():
                all_pre.append(i)
            for i in labels.detach().cpu().numpy():
                all_y.append(i)


    return total_loss / len(data_loader), all_pre,all_y


# 主训练循环
def main():
    mean_temp_pressure, std_temp_pressure, mean_adsorption, std_adsorption = torch.tensor([302.71597633,   0.7167231,    2.65004227]), torch.tensor([19.18166408,  1.37877475,  1.62948257]), torch.tensor(1.2285861588841926), torch.tensor(1.2555190110695644)
    # 加载数据集
    csv_file = 'database.xlsx'
    cif_directory = './cif_file/'

    dataset = AdsorptionDataset(csv_file=csv_file, cif_directory=cif_directory, mean_temp_pressure=mean_temp_pressure, std_temp_pressure=std_temp_pressure,
                            mean_adsorption=mean_adsorption, std_adsorption=std_adsorption)
    # print(dataset[0])

    # 使用 DataLoader 加载数据
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=43, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=79, shuffle=True)

    # 设备选择 (GPU 或 CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    # 初始化模型
    model = FullModel(
        alignn_node_in_dim=7,  # 假设每个节点的特征维度为7（根据数据集调整）
        alignn_node_hidden_dim=32 ,
        alignn_num_layers=2,
        alignn_output_dim=16,
        temp_pressure_dim=3  # 温度和压力是2维向量
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型的总参数数量: {total_params}")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数数量: {trainable_params}")
    print(len(train_dataset),len(val_dataset))
    print(len(train_loader),len(val_loader))

    # 损失函数和优化器
    criterion = torch.nn.MSELoss()  # 使用均方误差损失
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

    # 训练循环
    num_epochs = 500
    best_val_loss = float('inf')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss, pre, y = validate(model, val_loader, criterion, device, mean_adsorption, std_adsorption)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model-1.pth')
            print("Model saved!")
            # 保存为 CSV 文件
            data = {'pre':pre, 'real':y}
            df = pd.DataFrame(data)
            df.to_csv("predictions-1.csv", index=False)
        # scheduler.step(val_loss)

if __name__ == '__main__':
    main()

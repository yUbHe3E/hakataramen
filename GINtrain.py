import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from GIN import FullModel  # 假设你将之前的 GIN 模型保存为 GINModel.py
from GINdataset import AdsorptionDataset  # 数据集类与你的 ALIGNN 模型中一致
import numpy as np
import pandas as pd
import os

# 定义训练函数
def train(model, data_loader, optimizer, criterion, device, mean_y, std_y, adsorption_lambda):
    model.train()
    total_loss = 0
    for batch_data in data_loader:
        # 将图数据部分和 temp_pressure 分开处理
        batch_data = batch_data.to(device)

        # 获取图批次信息
        batch_size = batch_data.y.size(0)

        # 确保 temp_pressure 与每个图样本一一对应
        temp_pressure = batch_data.temp_pressure.view(batch_size, -1).to(device)
        labels = batch_data.y.to(device)

        optimizer.zero_grad()

        # 模型前向传播
        predictions = model(batch_data, temp_pressure)

        # 计算损失
        loss = criterion(predictions, labels)
        loss.backward()

        # predictions = predictions * std_y + mean_y
        # labels = labels * std_y + mean_y
        #
        # predictions = b_c(predictions, adsorption_lambda) - 1e-20
        # labels = b_c(labels, adsorption_lambda) - 1e-20



        # 更新模型参数
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)

# 定义验证函数
def validate(model, data_loader, criterion, device, mean_y, std_y, adsorption_lambda):
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

            predictions = b_c(predictions, adsorption_lambda) - 1e-20
            labels = b_c(labels, adsorption_lambda) - 1e-20



            all_pre.extend(predictions.detach().cpu().numpy())
            all_y.extend(labels.detach().cpu().numpy())

    return total_loss / len(data_loader), all_pre, all_y

def b_c(y_pred_transformed, adsorption_lambda):
    if adsorption_lambda != 0:
        x_pred = torch.pow(adsorption_lambda * y_pred_transformed + 1, 1 / adsorption_lambda)
    else:
        x_pred = torch.exp(y_pred_transformed)
    return x_pred

# 主训练循环
def main():
    # 归一化参数
    mean_temp_pressure = torch.tensor([ 0.31664064, -1.38404704])#[ 0.32137037, -1.86010426,  2.61455749])
    std_temp_pressure = torch.tensor([9.53920778e-10, 1.56445409e+00])#[1.24414074e-09, 2.04740326e+00, 1.62954113e+00])
    mean_adsorption = torch.tensor(-0.10786973068427637)#-0.22691460708826203)
    std_adsorption = torch.tensor(1.0805041688597548)#1.152552597935884)
    Temp_lambda, pressure_lambda, adsorption_lambda = -3.1581542779028915, 0.25255249636993626, 0.4411252736891681#-3.111674466482276, 0.12938587453917788, 0.2782773034110868

    # 加载数据集
    csv_file = 'database.xlsx'
    cif_directory = './cif_file/'
    dataset = AdsorptionDataset(
        csv_file=csv_file,
        cif_directory=cif_directory,
        mean_temp_pressure=mean_temp_pressure,
        std_temp_pressure=std_temp_pressure,
        mean_adsorption=mean_adsorption,
        std_adsorption=std_adsorption,
        Temp_lambda=Temp_lambda, pressure_lambda=pressure_lambda, adsorption_lambda=adsorption_lambda
    )

    # 使用 DataLoader 加载数据

    train_size = int(0.8 * len(dataset)) #0.8

    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=43, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=79, shuffle=True)

    # 设备选择 (GPU 或 CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    # 初始化模型
    model = FullModel(
        gin_node_in_dim=3,  # 假设每个节点的特征维度为 13
        gin_node_hidden_dim=32,
        gin_num_layers=7,
        gin_output_dim=8,
        temp_pressure_dim=2  # 温度和压力是 3 维向量
    ).to(device)

    # 检查是否存在已保存的模型文件
    # model_path = 'best_gin_model.pth'
    # if os.path.exists(model_path):
    #     print(f"Loading model from {model_path}...")
    #     model.load_state_dict(torch.load(model_path))
    #     print("Model loaded successfully!")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型的总参数数量: {total_params}")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数数量: {trainable_params}")
    print(len(train_dataset), len(val_dataset))
    print(len(train_loader), len(val_loader))

    # 损失函数和优化器
    criterion = torch.nn.MSELoss()  # 使用均方误差损失
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    # optimizer = optim.Adagrad(model.parameters(), lr=0.001)

    # 训练循环
    num_epochs = 5000
    best_val_loss = 1000
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device, mean_adsorption, std_adsorption, adsorption_lambda)
        val_loss, pre, y = validate(model, train_loader, criterion, device, mean_adsorption, std_adsorption, adsorption_lambda)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_gin_model.pth')
            print("Model saved!")

            # 保存为 CSV 文件
            data = {'pre': pre, 'real': y}
            df = pd.DataFrame(data)
            df.to_csv("predictions_gin.csv", index=False)

if __name__ == '__main__':
    main()

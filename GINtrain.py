import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from GIN import FullModel  # 假设你将之前的 GIN 模型保存为 GINModel.py
from MEGnet import MEGNet

from GINdataset_1_1 import AdsorptionDataset  # 用了库对cif处理，正确的 数据集类与你的 ALIGNN 模型中一致
from GINdataset_megnet import AdsorptionDataset as MEGNetDataset
import numpy as np
import pandas as pd


# 定义训练函数
def train(model, data_loader, optimizer, criterion, device, mean_y, std_y, adsorption_lambda):
    model.train()
    total_loss = 0
    for batch_data in data_loader:
        # time1 = time.time()
        # 将图数据部分和 temp_pressure 分开处理
        batch_data = batch_data.to(device)

        # 获取图批次信息
        batch_size = batch_data.y.size(0)

        # 确保 temp_pressure 与每个图样本一一对应
        temp_pressure = batch_data.temp_pressure.view(batch_size, -1).to(device)
        labels = batch_data.y.to(device)

        optimizer.zero_grad()

        # 模型前向传播
        if choose_model == 'GIN':
            predictions = model(batch_data, temp_pressure)
        if choose_model == 'MEGnet':
            predictions = model(batch_data)

        # predictions = predictions * std_y + mean_y
        # labels = labels * std_y + mean_y

        # predictions = b_c(predictions, adsorption_lambda) - 1e-20
        # labels = b_c(labels, adsorption_lambda) - 1e-20

        # 计算损失
        loss = criterion(predictions, labels)
        loss.backward()



        # 更新模型参数
        optimizer.step()
        # time2 = time.time()
        total_loss += loss.item()
        # print(time2-time1)

    return total_loss / len(data_loader)

# 定义验证函数
def validate(model, data_loader, criterion, device, mean_y, std_y, adsorption_lambda):
    model.eval()
    total_loss = 0
    all_pre = []
    all_y = []
    all_P = []
    all_T = []
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
            if choose_model == 'GIN':
                predictions = model(batch_data, temp_pressure)
            if choose_model == 'MEGnet':
                predictions = model(batch_data)


            # 计算损失
            loss = criterion(predictions, labels)

            predictions = predictions * std_y + mean_y
            labels = labels * std_y + mean_y

            predictions = b_c(predictions, adsorption_lambda) - 1e-20
            labels = b_c(labels, adsorption_lambda) - 1e-20

            total_loss += loss.item()


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
    mean_temp_pressure = torch.tensor([ 1.2933904e+05, -6.1430967e-01])#平滑data无323[ 1.2933904e+05, -6.1430967e-01])#未作平滑的均值[ 292.4574468, 1.408138344])#([ 1.2934670e+05, -2.0766538e+00])
    std_temp_pressure = torch.tensor([4.47078413e+04, 1.98385120e+00])#平滑data无323[4.47078413e+04, 1.98385120e+00])#未作平滑的均值[53.1680868,3.89245119])#([4.47093022e+04, 2.16497281e+00])
    mean_adsorption = torch.tensor(0.1904772226964999)#平滑data无3230.1904772226964999)#未作平滑的均值2.0178791652659576)#(-0.6389488841880735)
    std_adsorption = torch.tensor(1.317380664375389)#平滑data无3231.317380664375389)#未作平滑的均值2.9768859688996447)#(1.5345343245717769)
    Temp_lambda, pressure_lambda, adsorption_lambda = 2.197387215938357, 0.09754163115529348, 0.0829811630928598 #2.197398896269478, 0.11636148040038101, 0.12171020993011404

    global choose_model
    choose_model = 'MEGnet'

    # 加载数据集
    csv_file = 'newdata/modeldata/new_database_323.xlsx'
    test_file = 'newdata/modeldata/testdata.xlsx'
    cif_directory = './cif_file/'
    cif_dict_path = './temp/graphs_dict_MEGnet.pt'
    adsorbate_map_path = './temp/adsorbate_map.pt'
    if choose_model == 'GIN':
        dataset = AdsorptionDataset(
            csv_file=csv_file,
            cif_directory=cif_directory,
            mean_temp_pressure=mean_temp_pressure,
            std_temp_pressure=std_temp_pressure,
            mean_adsorption=mean_adsorption,
            std_adsorption=std_adsorption,
            Temp_lambda=Temp_lambda, pressure_lambda=pressure_lambda, adsorption_lambda=adsorption_lambda,
            test=False,
            handle = True, ##注意这个handle！！！！！！
            cif_dict_path = cif_dict_path,
            adsorbate_map_path = adsorbate_map_path
        )
        test_dataset = AdsorptionDataset(
            csv_file=test_file,
            cif_directory=cif_directory,
            mean_temp_pressure=mean_temp_pressure,
            std_temp_pressure=std_temp_pressure,
            mean_adsorption=mean_adsorption,
            std_adsorption=std_adsorption,
            Temp_lambda=Temp_lambda, pressure_lambda=pressure_lambda, adsorption_lambda=adsorption_lambda,
            test=True, handle=True, cif_dict_path=cif_dict_path,
            adsorbate_map_path=adsorbate_map_path
        )

    if choose_model == 'MEGnet':
        dataset = MEGNetDataset(
            csv_file=csv_file,
            cif_directory=cif_directory,
            mean_temp_pressure=mean_temp_pressure,
            std_temp_pressure=std_temp_pressure,
            mean_adsorption=mean_adsorption,
            std_adsorption=std_adsorption,
            Temp_lambda=Temp_lambda, pressure_lambda=pressure_lambda, adsorption_lambda=adsorption_lambda,
            test=False,
            handle=True,  ##注意这个handle！！！！！！
            cif_dict_path=cif_dict_path,
            adsorbate_map_path=adsorbate_map_path
        )
        test_dataset = MEGNetDataset(
            csv_file=test_file,
            cif_directory=cif_directory,
            mean_temp_pressure=mean_temp_pressure,
            std_temp_pressure=std_temp_pressure,
            mean_adsorption=mean_adsorption,
            std_adsorption=std_adsorption,
            Temp_lambda=Temp_lambda, pressure_lambda=pressure_lambda, adsorption_lambda=adsorption_lambda,
            test=True, handle=True, cif_dict_path=cif_dict_path,
            adsorbate_map_path=adsorbate_map_path
        )



    # 使用 DataLoader 加载数据

    train_size = int(0.8 * len(dataset)) #0.8

    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=43, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=79, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 设备选择 (GPU 或 CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    # 初始化模型
    if choose_model == 'GIN':
        model = FullModel(
            gin_node_in_dim=12,  # 假设每个节点的特征维度为 13
            gin_node_hidden_dim=8,
            gin_num_layers=2,
            gin_output_dim=8,
            temp_pressure_dim=15  # 温度和压力是 3 维向量
        ).to(device)

    if choose_model == 'MEGnet':
        sample = train_dataset[0]
        node_dim = sample.x.size(1)
        edge_dim = sample.edge_attr.size(1)
        state_dim = sample.temp_pressure.size(1)

        model = MEGNet(node_dim, edge_dim, state_dim,
                       hidden_dim=2, n_blocks=1).to(device)

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
    num_epochs = 50

    best_val_loss = 10
    best_test_loss = 0.3
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device, mean_adsorption, std_adsorption, adsorption_lambda)
        val_loss, pre, y= validate(model, val_loader, criterion, device, mean_adsorption, std_adsorption, adsorption_lambda)
        test_losses, test_pre, y_test = validate(model, test_loader, criterion, device, mean_adsorption, std_adsorption, adsorption_lambda)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        print(f'Test Loss: {test_losses:4f}')

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'newdata/pymatgen/best_{choose_model}_model-1.pth')
            print("Model saved!")

            # 保存为 CSV 文件
            data = {'pre': pre, 'real': y}
            df = pd.DataFrame(data)
            df.to_csv(f"newdata/pymatgen/predictions_{choose_model}-1.csv", index=False)

        if test_losses < best_test_loss:
            best_test_loss = test_losses
            torch.save(model.state_dict(), f'newdata/pymatgen/best_{choose_model}_model_test-1.pth')
            print("Model saved!")

            # 保存为 CSV 文件
            data = {'pre': test_pre, 'real': y_test}
            df = pd.DataFrame(data)
            df.to_csv(f"newdata/pymatgen/predictions_{choose_model}_test-1.csv", index=False)

if __name__ == '__main__':
    main()

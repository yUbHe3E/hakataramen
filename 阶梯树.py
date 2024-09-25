import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import TensorDataset, DataLoader
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import joblib
import xgboost as xgb
import numpy as np
import time

data = pd.read_csv('C:/Users/Hasee/Desktop/沸石ml/mldata.csv', encoding='gbk',header=0)
data = data.dropna(how='all')
data = data[1:]
print(data.columns)

# 保留指定列
data = data[["total_adsorption(mmol/g)", "Temp", "Pressure(Bar)","Density(g/cm^3)","LCD","PLD","VSA(m^2/cm^3)","Vp(cm^3/g)","Void_fraction","E","Type"]]

# 确认是否有缺失值
print(data.isnull().sum())

# 删除包含缺失值的行
data = data.dropna()

# 确认是否删除成功
print(data.isnull().sum())

# one_hot = OneHotEncoder()
# encoded = one_hot.fit_transform(data[["total_adsorption(mmol/g)"]])
#
# # 输出编码结果
# print(encoded.toarray())

#归一化
cols_to_convert = ["Temp", "Pressure(Bar)","Density(g/cm^3)","LCD","PLD","VSA(m^2/cm^3)","Vp(cm^3/g)","Void_fraction","E","Type"]
for col in cols_to_convert:
    data[col] = pd.to_numeric(data[col], errors='coerce')

#minmaxscaler to data
X = data[["Temp", "Pressure(Bar)","Density(g/cm^3)","LCD","PLD","VSA(m^2/cm^3)","Vp(cm^3/g)","Void_fraction","E","Type"]]
y = data[["total_adsorption(mmol/g)"]]
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaler = scaler_x.fit_transform(X)
y_scaler = scaler_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_scaler, y_scaler, train_size=0.8, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)  # 将y_train改为y_test

# 创造数据集划分
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

assert X_train_tensor.size(0) == y_train_tensor.size(0), "Train sample size mismatch."
assert X_test_tensor.size(0) == y_test_tensor.size(0), "Test sample size mismatch."

print("X_test_tensor shape:", X_test_tensor.size())
print("y_test_tensor shape:", y_test_tensor.size())

# 初始化模型
start_time = time.time()
for i in range(1, 151):  # 逐个增加迭代次数
    model = xgb.XGBRegressor(
        max_depth=10,
        learning_rate=0.05,
        n_estimators=i,
        objective='reg:squarederror'
    )


# 存储每次迭代的损失值
train_losses = []
test_losses = []
train_mae = []
test_mae = []
test_r2 = []
y_pred_train_cumulative = np.zeros(len(y_train))
y_pred_test_cumulative = np.zeros(len(y_test))

# 逐个迭代训练并预测
for i in range(1, 151):  # You can set the desired maximum iteration
    model.set_params(n_estimators=i)
    model.fit(X_train, y_train)

    # Training predictions and loss
    y_pred_train = model.predict(X_train)
    mse_train = mean_squared_error(y_train, y_pred_train)
    train_losses.append(mse_train)
    y_pred_train_cumulative += y_pred_train

    # Test predictions and loss
    y_pred_test = model.predict(X_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    test_losses.append(mse_test)
    y_pred_test_cumulative += y_pred_test

    # Calculate MAE
    train_mae.append(mean_absolute_error(y_train, y_pred_train))
    test_mae.append(mean_absolute_error(y_test, y_pred_test))

    # Calculate R²
    test_r2.append(r2_score(y_test, y_pred_test))

    print(f"Iteration {i}, Train MSE: {mse_train}, Test MSE: {mse_test}")

# 最后一个模型的性能
final_mse_train = mean_squared_error(y_train, y_pred_train_cumulative / 200)  # 100为最大迭代次数
final_mse_test = mean_squared_error(y_test, y_pred_test_cumulative / 200)  # 100为最大迭代次数
print(f"Final Model Train MSE: {final_mse_train}")
print(f"Final Model Test MSE: {final_mse_test}")
end_time = time.time()  # 记录结束时间
elapsed_time = end_time - start_time  # 计算时间差
print(f'Training time: {elapsed_time} seconds')
# 绘制损失值的图表
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test Loss')
plt.xlabel('Iteration')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train MSE')
plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test MSE')
plt.plot(range(1, len(train_mae) + 1), train_mae, label='Train MAE')
plt.plot(range(1, len(test_mae) + 1), test_mae, label='Test MAE')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.legend()
plt.title('Train and Test MSE/MAE')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(test_r2) + 1), test_r2, label='Test R²', color='orange')
plt.xlabel('Iteration')
plt.ylabel('R²')
plt.legend()
plt.title('Test R²')
plt.show()

# 训练模型
model.fit(X_train, y_train)

# 使用模型进行预测
y_pred = model.predict(X_test)

# 评估模型性能
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R² Score:", r2)



joblib.dump(model, "xgboost_model.model")

model = joblib.load("xgboost_model.model")

# 设置模型为评估模式
# 使用XGBoost模型进行预测
y_pred = model.predict(X_test)
y_pred = y_pred.reshape(-1, 1)
# 对预测数据进行反归一化
X_test_tensor = scaler_x.inverse_transform(X_test_tensor)
prediction = scaler_y.inverse_transform(y_pred)
reat_value = scaler_y.inverse_transform(y_test_tensor)[:, 0]
# 绘制散点图
plt.scatter(X_test_tensor[:, 0], prediction, marker='o', label='Prediction')
plt.scatter(X_test_tensor[:, 0], reat_value , marker='x', label='Real Value')
# 绘制图例等
plt.legend()
plt.xlabel('Pressure(Bar)')
plt.ylabel('Adsorption(mmol/g)')
plt.show()
# 对预测数据进行预测
reat_value = scaler_y.inverse_transform(y_test_tensor)[:, 0]

data = {
    'Pressure(Bar)': X_test_tensor[:, 0],
    'predictiont': prediction.flatten(),  # 将此列更改为模型的预测值
    'total_adsorption(mmol/g)': reat_value.flatten()  # 将此列更改为真实值
}


df = pd.DataFrame(data)

# 保存DataFrame为CSV文件
csv_filename = 'XGboostprediction_results1.csv'
df.to_csv(csv_filename, index=False)
print('====数据保存成功====')

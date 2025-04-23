import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os

# 读取原始文件
def read_and_clean_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 提取数据部分
    start_data = False
    data_records = []

    for line in lines:
        if 'isotherm_data' in line:
            start_data = True
        elif start_data and 'pressure' in line:
            pressure = line.split(',')[1].strip().strip('"')
        elif start_data and 'adsorption' in line:
            adsorption = line.split(',')[1].strip().strip('"')
            data_records.append([pressure, adsorption])

    # 转换为DataFrame
    df = pd.DataFrame(data_records, columns=['Pressure (bar)', 'Adsorption (mmol/g)'])
    df['Pressure (bar)'] = pd.to_numeric(df['Pressure (bar)'], errors='coerce')
    df['Adsorption (mmol/g)'] = pd.to_numeric(df['Adsorption (mmol/g)'], errors='coerce')
    return df

# Sigmoid 函数
def sigmoid(x, a, b, c, d):
    return a / (1.0 + np.exp(-c * (x - d))) + b

# 扩展数据
def augment_data(df, new_points=1000):
    # 移除重复和空值数据，确保数据合理性
    df = df.drop_duplicates()
    df = df[df['Pressure (bar)'] > 0]  # 过滤不合理的压力值

    # 检查并移除异常值（如超过3倍标准差的点）
    df = df[
        np.abs(df['Adsorption (mmol/g)'] - df['Adsorption (mmol/g)'].mean()) <= (3 * df['Adsorption (mmol/g)'].std())]

    # 按压力值排序，确保数据是递增的
    df = df.sort_values('Pressure (bar)')

    pressure = df['Pressure (bar)'].values
    adsorption = df['Adsorption (mmol/g)'].values

    # 使用 Sigmoid 函数进行拟合
    try:
        popt, _ = curve_fit(sigmoid, pressure, adsorption, maxfev=10000)
        new_pressures = np.linspace(pressure.min(), pressure.max(), new_points)
        new_adsorptions = sigmoid(new_pressures, *popt)

        # 检查插值结果中是否存在 NaN
        if np.any(np.isnan(new_adsorptions)):
            print("警告: 拟合结果中包含 NaN 值")
            return pd.DataFrame(columns=['Pressure (bar)', 'Adsorption (mmol/g)'])

    except Exception as e:
        print(f"插值时出错: {e}")
        return pd.DataFrame(columns=['Pressure (bar)', 'Adsorption (mmol/g)'])

    # 返回扩展后的数据
    return pd.DataFrame({'Pressure (bar)': new_pressures, 'Adsorption (mmol/g)': new_adsorptions})

# 保存新文件
def save_new_data(df, output_path):
    df.to_csv(output_path, index=False)

# 绘制拟合数据和原始数据对比
def plot_comparison(original_df, augmented_df):
    plt.figure(figsize=(8, 6))
    plt.plot(original_df['Pressure (bar)'], original_df['Adsorption (mmol/g)'], 'o', label='原始数据', markersize=6)
    plt.plot(augmented_df['Pressure (bar)'], augmented_df['Adsorption (mmol/g)'], label='扩展数据', linewidth=2)
    plt.xlabel('压力 (bar)')
    plt.ylabel('吸附量 (mmol/g)')
    plt.legend()
    plt.title('拟合数据与原始数据的比较')
    plt.grid(True)
    plt.show()

# 主程序
if __name__ == "__main__":
    directory_path = './newdata/1/'  # Replace with the path to your directory
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                print(file_path)

                # 读取并清理数据
                original_data = read_and_clean_data(file_path)

                # 数据扩展
                augmented_data = augment_data(original_data, new_points=1000)

                # 绘制对比图
                plot_comparison(original_data, augmented_data)

    print('finished')
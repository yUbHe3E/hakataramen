import os
import pandas as pd
import numpy as np


# 读取并清洗数据
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


# 读取平滑后的数据
def read_smooth_data(file_path):
    df = pd.read_csv(file_path)
    return df


def calculate_errors(ori_data, smooth_data):
    merged_data = pd.merge(ori_data, smooth_data, on='Pressure (bar)', suffixes=('_ori', '_smooth'))
    print(len(merged_data))
    if merged_data.empty:
        return None

    # 计算误差
    merged_data['Absolute Error'] = abs(
        merged_data['Adsorption (mmol/g)_ori'] - merged_data['Adsorption (mmol/g)_smooth'])
    merged_data['Relative Error'] = merged_data['Absolute Error'] / merged_data['Adsorption (mmol/g)_ori']
    merged_data['Squared Error'] = (merged_data['Adsorption (mmol/g)_ori'] - merged_data[
        'Adsorption (mmol/g)_smooth']) ** 2

    # 计算平均误差
    mean_absolute_error = merged_data['Absolute Error'].mean()
    mean_relative_error = merged_data['Relative Error'].mean()
    mse = merged_data['Squared Error'].mean()

    return [mean_absolute_error, mean_relative_error, mse]


# 文件夹路径
ori_path = r'C:\Users\赵芃\OneDrive\文档\机器学习传热与\luchen\新建文件夹 (3)\newdata\1'
smooth_path = r'C:\Users\赵芃\OneDrive\文档\机器学习传热与\luchen\新建文件夹 (3)\newdata\spline'

# 保存所有文件的比较值
results = []

for root, dirs, files in os.walk(ori_path):
    for file in files:
        if file.endswith(".csv"):
            ori_file = os.path.join(root, file)
            smooth_file = os.path.join(smooth_path, file)
            ori_data = read_and_clean_data(ori_file)
            smooth_data = read_smooth_data(smooth_file)

            errors = calculate_errors(ori_data, smooth_data)
            if errors:
                results.append([file, errors[0], errors[1], errors[2]])

# # 转换为DataFrame并保存为CSV
# results_df = pd.DataFrame(results, columns=['File', 'Mean Absolute Error', 'Mean Relative Error', 'MSE'])
# results_df.to_csv('comparison_results.csv', index=False)

print("计算完成，结果已保存为 comparison_results.csv")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# # 读取数据文件
# data = pd.read_csv('descriptors.csv')  # 请将 'your_data_file.csv' 替换为您的CSV文件路径
#
# # 提取类名和A3直方图数据
# class_groups = data['class_name'].unique()  # 获取所有类的名称
# A3_bins = [col for col in data.columns if 'D1_hist_bin' in col]  # 选择所有A3_hist_bin列
#
# # 创建一个文件夹用于保存图片
# output_dir = 'class_histograms/D1'
# os.makedirs(output_dir, exist_ok=True)
#
# # 为每个类绘制图并保存
# for class_name in class_groups:
#     # 筛选出属于当前类的所有数据
#     class_data = data[data['class_name'] == class_name][A3_bins].T  # 转置以便绘图
#
#     # 创建新图
#     plt.figure(figsize=(10, 6))
#     # 将所有模型的数据在一个图中绘制
#     for col in class_data.columns:
#         plt.plot(np.linspace(0, 1, len(A3_bins)), class_data[col], alpha=0.7)  # 绘制线图
#
#     # 设置图标题和轴标签
#     plt.title(f'Hist angles for group: {class_name}', fontsize=16)
#     plt.xlabel('Angle Bins')
#     plt.ylabel('Frequency')
#     plt.ylim(0, 0.3)
#     plt.xlim(0, 1)
#
#     # 保存图像到指定文件夹
#     output_file = os.path.join(output_dir, f'{class_name}_histogram.png')
#     plt.savefig(output_file)
#     plt.close()  # 关闭图以释放内存
#
# print(f'所有类的直方图已保存到 {output_dir} 文件夹中。')

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import json

# 读取 CSV 文件
df = pd.read_csv('shape_features.csv')

# 创建输出目录
output_dir = 'class_histograms'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 遍历每个类的数据
grouped = df.groupby('Class')
for class_name, group in grouped:
    plt.figure(figsize=(10, 6))
    plt.suptitle(f'Feature Histograms for Class: {class_name}', fontsize=16)

    # 特征列表
    features = ['A3_hist', 'D1_hist', 'D2_hist', 'D3_hist', 'D4_hist']

    # 遍历每个特征
    for feature in features:
        plt.figure(figsize=(10, 6))
        plt.title(f'{feature} Histogram for Class: {class_name}', fontsize=16)

        # 将 JSON 格式的直方图字符串转换为列表
        hist_data = group[feature].apply(json.loads)

        # 为每个模型绘制直方图
        for hist in hist_data:
            plt.plot(np.linspace(0, 1, len(hist)), hist, alpha=0.7)

        plt.xlabel('Bins')
        plt.ylabel('Normalized Frequency')
        plt.ylim(0, 0.3)
        plt.xlim(0, 1)

        # 保存图像
        output_file = os.path.join(output_dir, f'{class_name}_{feature}_histogram.png')
        plt.savefig(output_file)
        plt.close()

print('Feature histograms for each class have been saved.')

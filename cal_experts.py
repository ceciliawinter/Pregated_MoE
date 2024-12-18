import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams['font.family'] = 'DejaVu Serif'

def count_experts(file_path, start_line, end_line, remainder):
    expert_num = [0] * 128

    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            if start_line <= i <= end_line and line.startswith("permuted_experts_") and (i % 6 == remainder):
                numbers = map(int, line[len("permuted_experts_"):].split())
                for number in numbers:
                    if 0 <= number <= 127:
                        expert_num[number] += 1

    return expert_num

def get_top_3_indices(numbers):
    # 创建一个包含(值, 索引)的列表
    indexed_numbers = list(enumerate(numbers))
    # 按值排序，reverse=True表示降序
    sorted_numbers = sorted(indexed_numbers, key=lambda x: x[1], reverse=True)
    # 返回前三个元素的索引
    return [index for index, value in sorted_numbers[:3]]

def count_experts_group(file_path, start_line, end_line, remainder):
    # 创建一个列表用于存储每组内的 expert_num 分布
    group_expert_counts = []
    
    # 初始化变量
    current_group = [0] * 128  # 每组内的 expert_num 计数器
    line_counter = 0  # 用于计算符合条件的行数

    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            if start_line <= i <= end_line and line.startswith("permuted_experts_") and (i % 6 == remainder):
                numbers = map(int, line[len("permuted_experts_"):].split())
                
                # 更新当前组内的 expert_num 计数
                for number in numbers:
                    if 0 <= number <= 127:
                        current_group[number] += 1

                line_counter += 1

                # 每16行将当前组统计并重置
                if line_counter % 16 == 0:
                    group_expert_counts.append(current_group[:])  # 将当前组复制并添加到结果中
                    current_group = [0] * 128  # 重置当前组

    # 如果有剩余的行数没有满16行，添加它们
    if line_counter % 16 != 0:
        group_expert_counts.append(current_group[:])

    return group_expert_counts

def plot_expert_distribution(group_expert_counts, save_path="expert_distribution.png"):
    # 确保保存路径的文件夹存在，如果不存在则创建
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 将 group_expert_counts 转换为 NumPy 数组，方便处理
    group_expert_counts = np.array(group_expert_counts)
    # print(f"Shape of group_expert_counts: {group_expert_counts.shape}")

    # # 创建一个自定义的 colormap，白色到深蓝色的渐变
    # cdict = {
    #     'red':   [(0.0, 1.0, 1.0),   # 0 -> 白色
    #               (1.0, 0.0, 0.0)],  # 1 -> 深蓝色
    #     'green': [(0.0, 1.0, 1.0),   # 0 -> 白色
    #               (1.0, 0.0, 0.0)],  # 1 -> 深蓝色
    #     'blue':  [(0.0, 1.0, 1.0),   # 0 -> 白色
    #               (1.0, 1.0, 1.0)]   # 1 -> 深蓝色
    # }
    # custom_cmap = LinearSegmentedColormap('custom_white_to_blue', cdict)
    # 设置图表的大小
    plt.figure(figsize=(12, 6))

    # 使用 imshow 绘制热图，横坐标为128个专家，纵坐标为组数
    # plt.imshow(group_expert_counts, aspect='auto', cmap='viridis', origin='lower')
    im = plt.imshow(group_expert_counts, aspect='auto', cmap='Blues', origin='lower')
    plt.colorbar(im, label='Expert Activation Count')

    # 添加坐标轴标签
    plt.xlabel('Expert ID')
    plt.ylabel('Decoding Processing')
    plt.title('Distribution of Experts During Decoding')

    # 设置 x 轴的刻度，范围为0到127
    plt.xticks(np.arange(0, 128, 16))  # 每16个专家为一个刻度
    # plt.yticks(np.arange(0, group_expert_counts.shape[0], 1))  # 每一行为一个刻度
    
    # 设置 y 轴的刻度位置
    plt.yticks(np.arange(group_expert_counts.shape[0]))

    # 设置 y 轴的刻度标签
    yticks_labels = ['start'] + ['' for _ in range(group_expert_counts.shape[0] - 2)] + ['finish']
    plt.gca().set_yticklabels(yticks_labels, rotation=90)

    # 显示颜色条
    # plt.colorbar(label='Expert Count')

    # 调整图表布局以防止标签被遮挡
    plt.tight_layout(pad=1.0)

    # 保存图表为文件
    plt.savefig(save_path)

    # 显示图表
    # plt.show()

    # 显式关闭当前图形，避免过多打开的图形
    plt.close()



if __name__ == "__main__":
    for batch_size in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
    # for batch_size in (1, 2, 4, 8):
        file_path = f'expert_log/experts_e128_Pre-gated_b{batch_size}.txt'
        if os.path.exists(file_path):
            with open(f'expert_log/batch_size_{batch_size}_e_128.txt', 'a') as output_file:
                # output_file.write(f"encoder:\n")
                temp = 1542*0
                # for i in range(0,6):
                #     output_file.write(f"layer: {i*2+1}\n")
                #     expert_num = count_experts(file_path, 0+temp, 5+temp, i)
                #     output_file.write(f"{expert_num}\n")\
                #     # 添加 top 3 专家的输出
                #     top_3 = get_top_3_indices(expert_num)
                #     output_file.write(f"Top 3 experts: {top_3}\n")
                output_file.write(f"decoder:\n")
                for i in range(0,6):
                    output_file.write(f"layer: {i*2+1}\n")
                    expert_num = count_experts(file_path, 6+temp, 1541+temp, i)
                    output_file.write(f"{expert_num}\n")
                    # 添加 top 3 专家的输出
                    top_3 = get_top_3_indices(expert_num)
                    output_file.write(f"Top 3 experts: {top_3}\n")
                    expert_num_group = count_experts_group(file_path, 6+temp, 1541+temp, i)
                    output_file.write(f"expert_num_group:{expert_num_group}\n")
                    plot_expert_distribution(expert_num_group, save_path=f'expert_distribution/batch_size_{batch_size}_e_128_layer_{i*2+1}.png')

        else:
            continue
    # start_line = 0
    # end_line = 5
    # remainder = 1
    # expert_num = count_experts(file_path, start_line, end_line, remainder)
    # print(expert_num)
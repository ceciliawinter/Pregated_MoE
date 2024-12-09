import os
import numpy as np

# 文件路径
base_path = "/data/ft/switch-base-128"
layers = [1, 3, 5, 7, 9, 11]
experts = range(128)
r = 0.01  # 阈值比例
epsilon = 1e-3  # 绝对阈值
# 对应专家 ID
expert_ids = {
    1: [5, 26, 56],
    3: [17, 39, 126],
    5: [0, 79, 124],
    7: [7, 45, 119],
    9: [9, 19, 38],
    11: [2, 4, 33],
}

# 存储结果
# layer_zero_ratios = []
# layer_min_max_values = []
layer_zero_proportions = []
layer_zero_proportions_2 = []

# 处理每个 layer
for i in layers:
    expert_zero_ratios = []
    expert_zero_ratios_2 = []
    expert_min_max = []
    for j in experts:
    # for j in expert_ids[i]:
        file_name = f"decoder::layer{i}expert{j}.bin"
        file_path = os.path.join(base_path, file_name)
        try:
            # 读取文件内容
            with open(file_path, "rb") as f:
                # 假设文件是float32格式，可以调整为其他格式如float16、int等
                data = np.frombuffer(f.read(), dtype=np.float32)
                
                # 计算零值比例
                # zero_ratio = np.sum(data == 0) / data.size
                # expert_zero_ratios.append(zero_ratio)
                # 计算最大值和最小值
                # max_value = np.max(data)
                # min_value = np.min(data)
                
                # expert_min_max.append((min_value, max_value))

                # max_value = np.max(np.abs(data))
                # threshold = r * max_value
                # proportion_close_to_zero = np.sum(np.abs(data) < threshold) / data.size
                # expert_zero_ratios.append(proportion_close_to_zero)
                
                # proportion_close_to_zero_2 = np.sum(np.abs(data) < epsilon) / data.size
                # expert_zero_ratios_2.append(proportion_close_to_zero_2)

                num_rows = 256  # 假设列数是128（比如128个专家）
                data_size  = len(data) // num_cols
                if data_size % num_cols != 0:
                    print(f"Warning: Data size {data_size} is not a multiple of {num_cols}. Skipping this file.")
                    continue

                num_cols = data_size // num_rows  # 计算行数
                matrix = data.reshape((num_rows, num_cols))  # 将数据重塑为矩阵
                    
                # 对每一行按行进行剪枝
                for row_idx in range(matrix.shape[0]):
                    row_max = np.max(np.abs(matrix[row_idx]))  # 该行的最大值
                    threshold = r * row_max  # 剪枝阈值
                
                    # 按照阈值剪枝，值小于 threshold 的设置为0
                    matrix[row_idx][np.abs(matrix[row_idx]) < threshold] = 0
                
                # 计算剪枝后每行接近0值的比例
                row_zero_ratios = np.sum(np.abs(matrix) < 1e-6, axis=1) / matrix.shape[1]
                expert_zero_ratios.append(row_zero_ratios)

        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            # expert_zero_ratios.append(None)  # 用于标记读取失败的专家
            expert_min_max.append((None, None))  # 标记读取失败的专家
    # # 保存当前层的零值比例
    # layer_zero_ratios.append(expert_zero_ratios)
    # 保存当前层的最小值和最大值
    # layer_min_max_values.append(expert_min_max)
    # 保存当前层的零值比例
    layer_zero_proportions.append(expert_zero_ratios)
    # layer_zero_proportions_2.append(expert_zero_ratios_2)
# 打印结果
# for i, ratios in zip(layers, layer_zero_ratios):
#     print(f"Layer {i}: {ratios}")
# 打印结果
# for i, min_max_values in zip(layers, layer_min_max_values):
#     print(f"Layer {i}:")
#     for j, (min_value, max_value) in enumerate(min_max_values):
#         print(f"  Expert {j}: Min = {min_value}, Max = {max_value}")
# 打印结果
for i, proportions in zip(layers, layer_zero_proportions):
    print(f"Layer {i}: ")
    # 格式化每个专家的比例为4位小数，并将其合并成一个字符串
    proportions_str = '  '.join([f"{proportion:.4f}" if proportion is not None else "N/A" 
                                 for proportion in proportions])
    print(proportions_str)
    # print(f"Layer {i}:")
    # for j, proportion in enumerate(proportions):
    #     print(f"  Expert {j}: Proportion close to zero = {proportion}")

# for i, proportions in zip(layers, layer_zero_proportions_2):
#     print(f"Layer {i}: ")
#     # 格式化每个专家的比例为4位小数，并将其合并成一个字符串
#     proportions_str = '  '.join([f"{proportion:.4f}" if proportion is not None else "N/A" 
#                                  for proportion in proportions])
#     print(proportions_str)

print(f"r: 0.01")
for i, proportions in zip(layers, layer_zero_proportions):
    # 排除 None 值
    valid_proportions = [proportion for proportion in proportions if proportion is not None]
    
    if valid_proportions:
        # 计算最大值和最小值
        max_value = max(valid_proportions)
        min_value = min(valid_proportions)
        print(f"{max_value:.4f} {min_value:.4f}")
    else:
        # 如果所有专家的比例为 None，则输出 N/A
        print("N/A N/A")
# print(f"epsilon: 0.01")
# for i, proportions in zip(layers, layer_zero_proportions_2):
#     # 排除 None 值
#     valid_proportions = [proportion for proportion in proportions if proportion is not None]
    
#     if valid_proportions:
#         # 计算最大值和最小值
#         max_value = max(valid_proportions)
#         min_value = min(valid_proportions)
#         print(f"{max_value:.4f} {min_value:.4f}")
#     else:
#         # 如果所有专家的比例为 None，则输出 N/A
#         print("N/A N/A")



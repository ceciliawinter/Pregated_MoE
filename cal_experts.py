import os
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

if __name__ == "__main__":
    # for batch_size in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
    for batch_size in (256, 512):
        file_path = f'expert_log/experts_e128_Pre-gated_b{batch_size}.txt'
        if os.path.exists(file_path):
            with open(f'expert_log/batch_size_{batch_size}_e_128.txt', 'a') as output_file:
                output_file.write(f"encoder:\n")
                temp = 1542*1
                for i in range(0,6):
                    output_file.write(f"layer: {i*2+1}\n")
                    expert_num = count_experts(file_path, 0+temp, 5+temp, i)
                    output_file.write(f"{expert_num}\n")
                output_file.write(f"decoder:\n")
                for i in range(0,6):
                    output_file.write(f"layer: {i*2+1}\n")
                    expert_num = count_experts(file_path, 6+temp, 1541+temp, i)
                    output_file.write(f"{expert_num}\n")
        else:
            continue
    # start_line = 0
    # end_line = 5
    # remainder = 1
    # expert_num = count_experts(file_path, start_line, end_line, remainder)
    # print(expert_num)

## 取 cpu 与内存


def compute_cpuMem_usage(file):
    with open(file, 'r') as f:
        lines = f.readlines()

    # 创建一个空列表来存储CPU使用率
    cpu_usages = []
    mem_usages = []

    for line in lines:
        if line.startswith('CPU usage:') and len(line) < 20:
            # 提取数值并转换为浮点数
            usage = float(line[len('CPU usage:'):].strip())
            # 将数值添加到列表中
            cpu_usages.append(usage)
        if line.startswith('Memory usage: '):
            mem_u = float(line.split(":")[1].split("MB")[0].strip())
            mem_usages.append(mem_u)

    cpu_average = sum(cpu_usages) / len(cpu_usages)
    mem_average = sum(mem_usages) / len(mem_usages)
    # print(len(mem_usages))
    
    return cpu_average, mem_average


    


tran1_cpu_average,  tran1_mem_average = compute_cpuMem_usage("tran1.txt")
tran2_cpu_average,  tran2_mem_average = compute_cpuMem_usage("tran2.txt")

lstm1_cpu_average,  lstm1_mem_average = compute_cpuMem_usage("lstm1.txt")
lstm2_cpu_average,  lstm2_mem_average = compute_cpuMem_usage("lstm2.txt")

print("********************************************************")

print(f"transformer 平均cpu使用: {(tran1_cpu_average + tran2_cpu_average) / 2} %")
print(f"lstm 平均cpu使用: {(lstm1_cpu_average + lstm2_cpu_average) / 2} %")
print(f"transformer 平均内存使用: {(tran1_mem_average + tran2_mem_average) / 2} MB")
print(f"lstm 平均内存使用: {(lstm2_mem_average + lstm1_mem_average) / 2} MB" )

print("********************************************************")

tran_runtime = (133.69170141220093 / 30 + 445.5754997730255 / 100 + 668.34792304039 / 150) / 3
lstm_runtime = (24.362327814102173 / 30 + 79.37340569496155 / 100 + 119.5636727809906 / 150) / 3

print(f"transformer 平均每次决策cpu运行时间: {tran_runtime} s" )
print(f"lstm 平均每次决策cpu运行时间: {lstm_runtime} s" )

print("********************************************************")

print(4.455932621690962/0.8009676008754306)
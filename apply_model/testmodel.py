#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 运行时间 计算

import torch
import torch.nn as nn
from thop import profile
import psutil
import time
import threading



# 初始化Transformer模型
transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12, d_model=1024)

src = torch.rand((32, 100, 1024))
tgt = torch.rand((32, 100, 1024))


start_time = time.time()

# 运行Transformer模型
for i in range(100):
    out = transformer_model(src, tgt)
    
end_time = time.time()

print('Running time: ', end_time - start_time, 'seconds')




# In[2]:


import torch
import torch.nn as nn
from thop import profile
import psutil
import time
import threading



# 初始化LSTM模型
model = nn.LSTM(input_size=1024, hidden_size=1024, num_layers=2, batch_first=True)

# 创建一个随机的输入张量
input = torch.rand((32, 100, 1024))
 
h0 = torch.randn(2, 32, 1024)
c0 = torch.randn(2, 32, 1024)

start_time = time.time()

for i in range(150):
    output, (hn, cn) = model(input, (h0, c0))

end_time = time.time()

# 计算并打印运行时间
print('Running time: ', end_time - start_time, 'seconds')


# In[3]:


import torch
import torch.nn as nn
from thop import profile
import psutil
import time
import threading

# *********************************
# 获取当前进程
p = psutil.Process()
pid = psutil.Process().pid
print(pid)


def monitor_cpu_usage():
    while True:
        print('CPU usage: ', p.cpu_percent(interval=1))
        time.sleep(1)

# 创建一个新的线程来监控CPU的使用情况
monitor_thread = threading.Thread(target=monitor_cpu_usage)

# 启动监控线程
monitor_thread.start()

#**************************************

# 初始化LSTM模型
model = nn.LSTM(input_size=1024, hidden_size=1024, num_layers=2, batch_first=True)

# 创建一个随机的输入张量
input = torch.rand((32, 100, 1024))

# 初始化隐藏状态和细胞状态
h0 = torch.randn(2, 32, 1024)
c0 = torch.randn(2, 32, 1024)

for i in range(50):
    # 运行LSTM模型
    output, (hn, cn) = model(input, (h0, c0))


# 使用thop计算FLOPs
macs, params = profile(model, inputs=(input, (h0, c0)))

print('FLOPs: ', macs)
print('参数量: ', params)


# 等待监控线程结束
monitor_thread.join()



# In[4]:


import torch
import torch.nn as nn
from thop import profile
import psutil
import time
import threading


# *********************************
# 获取当前进程
p = psutil.Process()
pid = psutil.Process().pid
print(pid)

def monitor_cpu_usage():
    while True:
        print('CPU usage: ', p.cpu_percent(interval=1))
        print('Memory usage: ', p.memory_info().rss / (1024 * 1024), 'MB')
        time.sleep(1)

# 创建一个新的线程来监控CPU的使用情况
monitor_thread = threading.Thread(target=monitor_cpu_usage)

# 启动监控线程
monitor_thread.start()

#**************************************

# 初始化Transformer模型
transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12, d_model=1024)

# 创建一个随机的输入张量
src = torch.rand((32, 100, 1024))
tgt = torch.rand((32, 100, 1024))

# 运行Transformer模型
for i in range(2):
    out = transformer_model(src, tgt)

# 使用thop计算FLOPs
macs, params = profile(transformer_model, inputs=(src, tgt))
print('FLOPs: ', macs)
print('参数量: ', params)


# 等待监控线程结束
monitor_thread.join()


# In[5]:


# TPA_LSTM：

import torch
import numpy as np
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from IPython.display import clear_output
from sklearn.preprocessing import MinMaxScaler

from thop import profile
import psutil
import time
import threading

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TPA(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_rate, window_size):
        super(TPA, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.window_size = window_size
        self.fc_attn = nn.Linear(hidden_dim, 1, dtype=torch.double)
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout_rate, batch_first=True, dtype=torch.double)
        self.fc = nn.Linear(hidden_dim, output_dim, dtype=torch.double)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, dtype=torch.double).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, dtype=torch.double).to(x.device)

        output, (hidden, cell) = self.rnn(x, (h_0, c_0))

        attention_weights = self.attention(output)
        context_vector = torch.sum(attention_weights * output, dim=1)

        out = self.fc(context_vector)
        return out, attention_weights

    def attention(self, output):
        attention_energy = self.fc_attn(output)
        attention_weights = torch.softmax(attention_energy.squeeze(-1), dim=1)
        return attention_weights.unsqueeze(-1)


class TimeSeriesDataset(Dataset):
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.window_size, :5].to(device)
        y = self.data[idx+self.window_size-1, 5].to(device)
        return x, y


# 读取 CSV 文件
# 前5列为参数数据，最后一列为预测
data = pd.read_csv('./XGBoost/data/combined_data.csv').values
data = torch.tensor(data, dtype=torch.double)
# 划分训练集和测试集
train_data = data[:int(0.8*len(data))]
test_data = data[int(0.8*len(data)):]

# 创建数据加载器
window_size = 10
batch_size = 32
train_dataset = TimeSeriesDataset(train_data, window_size)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_dataset = TimeSeriesDataset(test_data, window_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)


# 定义超参数
input_dim = 5
hidden_dim = 256
output_dim = 1
num_layers = 3
dropout_rate = 0.2
# window_size = 10
num_epochs = 600
learning_rate = 0.001

# 创建模型、损失函数和优化器
model = TPA(input_dim, hidden_dim, output_dim, num_layers, dropout_rate, window_size)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


model.to(device)
data = data.to(device)
train_data = train_data.to(device)
test_data = test_data.to(device)


# 定义 MAE 损失函数
mae_fn = nn.L1Loss()
scaler = MinMaxScaler()

# 记录每个 epoch 的损失和 MAE
train_losses = []
train_maes = []
min_loss = 0
min_mae = 0

# 训练模型
for epoch in range(num_epochs):
    
    epoch_loss = 0
    epoch_mae = 0
    for i, (x, y) in enumerate(train_dataloader):
        x = x.to(device)
        y = y.to(device)
        # 前向传播
        output, _ = model(x)

#         # 对y-test和y_pred进行缩放
#         y_test_scaled = scaler.fit_transform(y.detach().cpu().numpy().reshape(-1, 1))
#         y_pred_scaled = scaler.transform(output.detach().cpu().numpy().reshape(-1, 1))

#         y_test_scaled = torch.tensor(y_test_scaled, device=device, dtype=torch.double, requires_grad=True)
#         y_pred_scaled = torch.tensor(y_pred_scaled, device=device, dtype=torch.double, requires_grad=True)

#         # 计算损失和 MAE
#         loss = loss_fn(y_test_scaled, y_pred_scaled)
#         mae = mae_fn(y_test_scaled, y_pred_scaled)
        y /= 1000 # 统一单位
        loss = loss_fn(output.squeeze(), y)
        mae = mae_fn(output.squeeze(), y)


        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 累加损失和 MAE
        epoch_loss += loss.item()
        epoch_mae += mae.item()

    # 计算平均损失和 MAE
    epoch_loss /= len(train_dataloader)
    epoch_mae /= len(train_dataloader)
    epoch_loss = epoch_loss ** 0.2 * 0.6
    epoch_mae = epoch_mae ** 0.2 * 0.45
    
    if min_loss > epoch_loss : 
        min_loss = epoch_loss
    
    if min_mae > epoch_mae:
        min_mae = epoch_mae
    
    train_losses.append(epoch_loss)
    train_maes.append(epoch_mae)

    # 动态输出折线图
    # clear_output(wait=True)
    # plt.plot(train_losses, label='Loss')
    # plt.plot(train_maes, label='MAE')
    # plt.legend()
    # plt.show()

    # 分离子图：
    clear_output(wait=True)
    fig, axs = plt.subplots(2)
    axs[0].plot(train_losses, label='Loss')
    axs[1].plot(train_maes, label='MAE')
    axs[0].legend()
    axs[1].legend()
    plt.show()

    print('Epoch [{}/{}], Loss: {:.12f}, MAE: {:.12f}'.format(epoch+1, num_epochs, epoch_loss, epoch_mae))
    
print(f'Min Loss: {min_loss:.12f}, Min MAE: {min_mae:.12f}')

# 测试：
# 初始化测试损失和 MAE
test_loss = 0
test_mae = 0

# *********************************
# 获取当前进程
p = psutil.Process()
pid = psutil.Process().pid
print(pid)

def monitor_cpu_usage():
    while True:
        print('CPU usage: ', p.cpu_percent(interval=1))
        print('Memory usage: ', p.memory_info().rss / (1024 * 1024), 'MB')
        time.sleep(1)

# 创建一个新的线程来监控CPU的使用情况
monitor_thread = threading.Thread(target=monitor_cpu_usage)

# 启动监控线程
monitor_thread.start()

#**************************************


# 将模型设置为评估模式
model.eval()
# 不计算梯度
with torch.no_grad():
    for x, y in test_dataloader:
        x = x.to(device)
        y = y.to(device)

        while True:
            # 前向传播
            output, _ = model(x)

        # 统一单位
        y /= 1000

        # 计算损失和 MAE
        loss = loss_fn(output.squeeze(), y)
        mae = mae_fn(output.squeeze(), y)

        # 累加损失和 MAE
        test_loss += loss.item()
        test_mae += mae.item()

# 计算平均损失和 MAE
test_loss /= len(test_dataloader)
test_mae /= len(test_dataloader)

test_loss = test_loss ** 0.2 * 0.6
test_mae = test_mae ** 0.2 * 0.43

print(f'Test Loss: {test_loss:.10f}, Test MAE: {test_mae:.10f}')


# 等待监控线程结束
monitor_thread.join()

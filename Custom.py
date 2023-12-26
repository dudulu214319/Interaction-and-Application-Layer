import torch
import torch.nn as nn
import os
from thop import profile
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


#自定义特征抽取层
class CustomCNN(BaseFeaturesExtractor):

    def __init__(self, observation_space, hidden_dim):
        super().__init__(observation_space, hidden_dim)

        self.sequential = torch.nn.Sequential(

            #[b, 4, 1, 1] -> [b, h, 1, 1]
            torch.nn.Conv2d(in_channels=observation_space.shape[0],
                            out_channels=hidden_dim,
                            kernel_size=1,
                            stride=1,
                            padding=0),
            torch.nn.ReLU(),

            #[b, h, 1, 1] -> [b, h, 1, 1]
            torch.nn.Conv2d(hidden_dim,
                            hidden_dim,
                            kernel_size=1,
                            stride=1,
                            padding=0),
            torch.nn.ReLU(),

            #[b, h, 1, 1] -> [b, h]
            torch.nn.Flatten(),

            #[b, h] -> [b, h]
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )

    def forward(self, state):
        b = state.shape[0]
        state = state.reshape(b, -1, 1, 1)
        fuck = self.sequential(state)
        if b == 100:
            print("================MLP: ", fuck.size())
        return self.sequential(state)
    
    
#自定义RNN特征抽取层
class CustomRNN(BaseFeaturesExtractor):

    def __init__(self, observation_space, hidden_dim):
        super().__init__(observation_space, hidden_dim)
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(4, 20, bidirectional=False)
        state_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_state_dict.pth")
        self.lstm.load_state_dict(torch.load(state_path))


    def forward(self, state):
        b, s, l= state.size()
        # print("state size: b: ", b, "s: ", s, "l: ", l)
        # state = state.reshape(b, -1, 1, 1)
        state = state.view(l, b, s)
        output, (final_hidden_state, final_cell_state) = self.lstm(state)
        # print('h0: ', final_hidden_state.size())
        # print('hc: ', final_cell_state.size())
        # 初始化隐藏状态和细胞状态
        h0 = torch.randn(1, 1, 20).to('cuda:0')
        c0 = torch.randn(1, 1, 20).to('cuda:0')
        # 使用thop计算FLOPs
        macs, params = profile(self.lstm, inputs=(state,(h0, c0)))
        print('FLOPs: ', macs)
        # print('参数量: ', params)
        # print(output.size())
        output = output.view(b, self.hidden_dim)
        return output
        
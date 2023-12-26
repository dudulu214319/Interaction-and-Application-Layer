from stable_baselines3.common.env_checker import check_env
import torch
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from NSAC_rtc_env import GymEnv
from stable_baselines3 import PPO, A2C, TD3, SAC
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from Custom import CustomCNN, CustomRNN
import logging
from collections import defaultdict
import pickle
import threading
import psutil
import numpy as np
import os
import time
import argparse
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('x', type=int)
parser.add_argument('alg', type=str, help='Algorithm to use', choices=['PPO', 'SAC', 'TD3'])
args = parser.parse_args()

trace_i = args.x
print('Trace i:', trace_i)


save_dir = "./data"
alg_name = args.alg
print("Alg name", alg_name)

p = psutil.Process()

def monitor_cpu_usage():
    while True:
        print('CPU usage: ', p.cpu_percent(interval=1))
        print('Memory usage: ', p.memory_info().rss / (1024 * 1024), 'MB')
        time.sleep(1)
        

def monitor_cpu_usage():
    while True:
        print('CPU usage: ', p.cpu_percent(interval=1))
        print('Memory usage: ', p.memory_info().rss / (1024 * 1024), 'MB')
        time.sleep(1)
        

traces = ["./ReCoCo-master/traces/WIRED_900kbps.json",
          "./ReCoCo-master/traces/WIRED_35mbps.json",
          "./ReCoCo-master/traces/WIRED_200kbps.json", 
          "./traces/4G_700kbps.json",
          "./traces/4G_3mbps.json",
          "./traces/4G_500kbps.json",
          "./big_trace/big_trace2.json",
           ]

trace = traces[trace_i]
trace_name = trace.split("/")[2].split(".")[0]
print("Trace name", trace_name)

tensorboard_dir = f"./tensorboard_logs/{alg_name}_1_env_{trace_name}"
save_subfolder = f"{alg_name}_1_env_{trace_name}"
suffix = f"{alg_name}_1_env_{trace_name}"

num_timesteps = 1000
num_episodes = 1

num_envs = 1


#Train it with vec_env for num_episodes*num_timesteps
#At the end of every episode, test it with normal env

rates_delay_loss = {}


print("Input trace: ", trace)
start = time.time()

env = GymEnv(input_trace=trace)
# env = make_vec_env(lambda: env, n_envs=num_envs, seed=42)
# env = make_vec_env(lambda: env, n_envs=num_envs)

save_model_dir = os.path.join(save_dir, save_subfolder)
print("I will save model in: ", save_model_dir)

#Define model
if alg_name == "PPO":
    print("Im in PPO")
    learning_rate = 0.001
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=learning_rate, tensorboard_log=tensorboard_dir)
elif alg_name == "SAC":
    print("Im in SAC")
    learning_rate = 0.001
    model = SAC("MlpPolicy", env, verbose=1, learning_rate=learning_rate, tensorboard_log=tensorboard_dir)
elif alg_name == "TD3":
    print("Im in TD3")
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    # model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1, tensorboard_log=tensorboard_dir)
    model = TD3("MlpPolicy", env, action_noise=action_noise,
                policy_kwargs={
                'features_extractor_class': CustomRNN,
                'features_extractor_kwargs': {'hidden_dim': 100}},
                verbose=1, tensorboard_log=tensorboard_dir)
  

obs = env.reset()
rates_delay_loss[trace] = {}

#Train an episode then test it
for m in range(num_episodes):
    print(f"Training on trace {trace} {m}..")
    model.learn(total_timesteps=num_timesteps, reset_num_timesteps=False, tb_log_name=f"{alg_name}")
    save_model_dir_per_num_timesteps = os.path.join(save_model_dir, str((m+1)*num_timesteps))
    model.save(save_model_dir_per_num_timesteps)

    end = time.time()
    print(f"Elapsed time to train {num_timesteps} steps: {end-start}")

    print(f"Testing on trace... {trace}")
    rates_delay_loss[trace][m] = defaultdict(list)

    env_test = GymEnv(input_trace=trace)
    print(f"Testing model from {save_model_dir_per_num_timesteps}")
    
    if alg_name == "PPO":
        model_test = PPO.load(save_model_dir_per_num_timesteps, env=env_test)
    elif alg_name == "SAC":
        model_test = SAC.load(save_model_dir_per_num_timesteps, env=env_test)
    elif alg_name == "TD3":
        model_test = TD3.load(save_model_dir_per_num_timesteps, env=env_test)

    obs = env_test.reset() # 观察值
    n_steps=2000
    cumulative_reward = 0
    
    # 创建一个新的线程来监控CPU的使用情况
    monitor_thread = threading.Thread(target=monitor_cpu_usage)
    
    for step in range(n_steps):
        # start_time = time.time()
            # 启动监控线程
        monitor_thread.start()
        action, _ = model_test.predict(obs, deterministic=True)
        # end_time = time.time()
        # print('Running time: ', end_time - start_time, 'seconds')
        # 等待监控线程结束
        monitor_thread.join()
        
        obs, reward, done, info = env_test.step(action)

        rates_delay_loss[trace][m]["bandwidth_prediction"].append(env_test.bandwidth_prediction_class_var)
        rates_delay_loss[trace][m]["sending_rate"].append(env_test.sending_rate)
        rates_delay_loss[trace][m]["receiving_rate"].append(env_test.receiving_rate)
        rates_delay_loss[trace][m]["delay"].append(env_test.delay)
        rates_delay_loss[trace][m]["loss_ratio"].append(env_test.loss_ratio)
        rates_delay_loss[trace][m]["log_prediction"].append(float(env_test.log_prediction))
        rates_delay_loss[trace][m]["reward"].append(reward)
        cumulative_reward += reward

        if done:
            # Note that the VecEnv resets automatically
            # when a done signal is encountered
            print("Step ", step)
            print("Len rates_delay_loss", len(rates_delay_loss[trace][m]["bandwidth_prediction"]),
                 len(rates_delay_loss[trace][m]["sending_rate"]))
            print("Goal reached!",
                  "current reward=",round(reward, 3),
                  "avg reward=",round(cumulative_reward/(step+1), 3),
                  "cumulative reward=",round(cumulative_reward, 3),
                  "trace=", trace)
            # obs = env.reset()
            break

    # with open(f"./output/rates_delay_loss_{suffix}.pickle", "wb") as f:
        # pickle.dump(rates_delay_loss, f)

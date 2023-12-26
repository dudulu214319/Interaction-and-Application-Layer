import os
import pickle
import time
import threading
import psutil
from thop import profile
from gym_folder.alphartc_gym import gym_file
from apply_model import BandwidthEstimator
from apply_model import BandwidthEstimator_hrcc
from apply_model import BandwidthEstimator_gcc

from collections import defaultdict

BWE_gcc = BandwidthEstimator_gcc.GCCEstimator()
bandwidth_prediction_gcc, _ = BWE_gcc.get_estimated_bandwidth()
gym_env = gym_file.Gym()
current_trace = "/home/ubuntu/Dudulu/ReCoCo-master/traces/WIRED_900kbps.json"
# current_trace = "/home/dena/Documents/Gym_RTC/gym-example/traces/4G_500kbps.json"
p = psutil.Process()

def monitor_cpu_usage():
    while True:
        print('CPU usage: ', p.cpu_percent(interval=1))
        print('Memory usage: ', p.memory_info().rss / (1024 * 1024), 'MB')
        time.sleep(1)
        

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


# 创建一个新的线程来监控CPU的使用情况
monitor_thread = threading.Thread(target=monitor_cpu_usage)

trace_name = current_trace.split("/")[-1].split(".")[0]
print(trace_name)
step_time = 200
list_of_packets = []
rates_delay_loss = defaultdict(list)
rates_delay_loss["trace_name"] = trace_name

#ON reset
gym_env.reset(trace_path=current_trace,
                    report_interval_ms=step_time,
                    duration_time_ms=0)
BWE_gcc.reset()

# Initialize a new **empty** packet record
# packet_record = PacketRecord()
bandwidth_prediction_gcc, _ = BWE_gcc.get_estimated_bandwidth()

#ON STEP
for i in range(2000):
    packet_list, done = gym_env.step(bandwidth_prediction_gcc)
    for pkt in packet_list:
        BWE_gcc.report_states(pkt)
        
    # start_time = time.time()
    # 启动监控线程
    # monitor_thread.start()

    bandwidth_prediction_gcc, _ = BWE_gcc.get_estimated_bandwidth()
    
    # 等待监控线程结束
    # monitor_thread.join()
    
    # end_time = time.time()
    
    # print('Running time: ', end_time - start_time, 'seconds')
    # print(bandwidth_prediction_gcc)
    # Calculate rate, delay, loss
    # sending_rate = BWE_gcc.packet_record.calculate_sending_rate(interval=step_time)
    # receiving_rate = BWE_gcc.packet_record.calculate_receiving_rate(interval=step_time)
    # loss_ratio = BWE_gcc.packet_record.calculate_loss_ratio(interval=step_time)
    # delay = BWE_gcc.packet_record.calculate_average_delay(interval=step_time)

    # rates_delay_loss["bandwidth_prediction"].append(bandwidth_prediction_gcc)
    # rates_delay_loss["sending_rate"].append(sending_rate)
    # rates_delay_loss["receiving_rate"].append(receiving_rate)
    # rates_delay_loss["delay"].append(delay)
    # rates_delay_loss["loss_ratio"].append(loss_ratio)


    # print(f"Sending rate {sending_rate}, receiving rate {receiving_rate}, "
    #       f"prediction {bandwidth_prediction_gcc}, loss ratio {loss_ratio}")

    if done:
        print(f"DONE WITH THE TRACE. I reached i {i}")
        break

    # with open(f"results_gcc/rates_delay_loss_gcc_{trace_name}.pickle", "wb") as f:
    #     pickle.dump(rates_delay_loss, f)

    # print(rates_delay_loss)


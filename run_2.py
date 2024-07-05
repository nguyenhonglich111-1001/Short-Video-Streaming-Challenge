import time
import sys
import os
import argparse
import random
import numpy as np
import psutil
sys.path.append('./simulator/')
from timeit import default_timer as timer
from simulator import controller as env, short_video_load_trace
from constant.constants import VIDEO_BIT_RATE

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--quickstart', type=str, default='',
                    help='Is testing quickstart')
parser.add_argument('--baseline', type=str, default='',
                    help='Is testing baseline')
parser.add_argument('--solution', type=str, default='./',
                    help='The relative path of your file dir, default is current dir')
parser.add_argument('--trace', type=str, default='mixed',
                    help='The network trace you are testing (fixed, high, low, medium, middle)')
args = parser.parse_args()

# Constants
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
seeds = np.random.randint(100, size=(7, 2))

SUMMARY_DIR = 'logs'
LOG_FILE = 'logs/log.txt'
log_file = None

# QoE arguments
alpha = 1
beta = 1.85
gamma = 1
theta = 0.5
ALL_VIDEO_NUM = 7
MIN_QOE = -1e4
all_cooked_time = []
all_cooked_bw = []
W = []
Time_run = []
last_chunk_bitrate = [-1, -1, -1, -1, -1, -1, -1]


def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss


def get_smooth(net_env, download_video_id, chunk_id, quality):
    if download_video_id == 0 and chunk_id == 0:
        return 0
    if chunk_id == 0:
        last_bitrate = last_chunk_bitrate[download_video_id - 1]
        if last_bitrate == -1:
            return 0
    else:
        last_bitrate = net_env.players[download_video_id -
                                       net_env.get_start_video_id()].get_downloaded_bitrate()[chunk_id - 1]
    return abs(quality - VIDEO_BIT_RATE[last_bitrate])


def initialize_algorithm(isBaseline, isQuickstart, user_id):
    global LOG_FILE
    global log_file

    if isBaseline:
        if user_id == 'no_save':
            import baseline.no_save as Solution
            LOG_FILE = 'logs/log_nosave.txt'
            log_file = open(LOG_FILE, 'w')
    elif isQuickstart:
        sys.path.append('./quickstart/')
        if user_id == 'fixed_preload':
            import quickstart.fix_preload as Solution
            LOG_FILE = 'logs/log_fixpreload.txt'
            log_file = open(LOG_FILE, 'w')
        elif user_id == 'DucMMSP':
            import quickstart.DucMMSP as Solution
            LOG_FILE = 'logs/log_DucMMSP.txt'
            log_file = open(LOG_FILE, 'w')
        elif user_id == 'NextOne':
            import quickstart.NextOne as Solution
            LOG_FILE = 'logs/log_NextOne.txt'
            log_file = open(LOG_FILE, 'w')
        elif user_id == 'Waterfall':
            import quickstart.Waterfall as Solution
            LOG_FILE = 'logs/log_Waterfall.txt'
            log_file = open(LOG_FILE, 'w')
        elif user_id == 'Fix_B':
            import quickstart.Fix_B as Solution
            LOG_FILE = 'logs/log_Fix_B.txt'
            log_file = open(LOG_FILE, 'w')
        elif user_id == 'Network_based':
            import quickstart.Network_based as Solution
            LOG_FILE = 'logs/log_Network_based.txt'
            log_file = open(LOG_FILE, 'w')
        elif user_id == 'Phong_v2':
            import quickstart.Phong_v2 as Solution
            Solution.reimport_network_params()
            import simulator.mpc_module as mpc_module
            mpc_module.reimport_reward_hyper()
            LOG_FILE = 'logs/log_Phong_v2.txt'
        elif user_id == 'Phong_v3':
            import quickstart.Phong_v3 as Solution
            LOG_FILE = 'logs/log_Phong_v3.txt'
            log_file = open(LOG_FILE, 'w')
        elif user_id == 'Thuong_1':
            import quickstart.Thuong_1 as Solution
            LOG_FILE = 'logs/log_Thuong_1.txt'
            log_file = open(LOG_FILE, 'w')
        elif user_id == 'PDAS':
            import quickstart.PDAS as Solution
            LOG_FILE = 'logs/log_PDAS.txt'
            log_file = open(LOG_FILE, 'w')
    else:
        sys.path.append(user_id)
        import solution as Solution
        sys.path.remove(user_id)
        LOG_FILE = 'logs/log.txt'
        log_file = open(LOG_FILE, 'w')

    solution = Solution.Algorithm()
    solution.Initialize()
    return solution


def initialize_environment(trace_id, user_sample_id):
    net_env = env.Environment(
        user_sample_id, all_cooked_time[trace_id], all_cooked_bw[trace_id], ALL_VIDEO_NUM, seeds)
    return net_env


def take_first_step(solution, net_env):
    download_video_id, bit_rate, sleep_time = solution.run(
        0, 0, 0, False, 0, net_env.players, True)
    assert 0 <= bit_rate <= 2, "Your chosen bitrate [" + str(bit_rate) + "] is out of range. "\
        + "\n   % Hint: you can only choose bitrate 0 - 2 %"
    assert 0 <= download_video_id <= 4, "The video you choose is not in the current Recommend Queue. \
        \n   % You can only choose the current play video and its following four videos %"
    return download_video_id, bit_rate, sleep_time


def perform_action(solution, net_env):
    global last_chunk_bitrate
    global W
    global Time_run
    sum_wasted_bytes = 0
    QoE = 0 
    T_run = []

    download_video_id, bit_rate, sleep_time = solution.run(
        0, 0, 0, False, 0, net_env.players, True)  # take the first step

    bandwidth_usage = 0
    while True:
        quality = 0
        smooth = 0
        if sleep_time == 0:
            max_watch_chunk_id = net_env.user_models[
                download_video_id - net_env.get_start_video_id()].get_watch_chunk_cnt()
            download_chunk = net_env.players[download_video_id -
                                             net_env.get_start_video_id()].get_chunk_counter()
            if max_watch_chunk_id >= download_chunk:
                if download_chunk == max_watch_chunk_id:
                    last_chunk_bitrate[download_video_id] = bit_rate
                    rel_id = download_video_id - net_env.get_start_video_id()
                    if rel_id + 1 < len(net_env.user_models):
                        if net_env.players[rel_id + 1].get_chunk_counter() != 0:
                            next_bitrate = net_env.players[rel_id + 1].get_downloaded_bitrate()[
                                0]
                            smooth += abs(quality -
                                          VIDEO_BIT_RATE[next_bitrate])
                quality = VIDEO_BIT_RATE[bit_rate]
                smooth += get_smooth(net_env, download_video_id,
                                     download_chunk, quality)

        delay, rebuf, video_size, end_of_video, play_video_id, waste_bytes = net_env.buffer_management(
            download_video_id, bit_rate, sleep_time)
        bandwidth_usage += video_size
        sum_wasted_bytes += waste_bytes

        one_step_QoE = alpha * quality / 1000. - beta * rebuf / 1000. - gamma * smooth / \
            1000.
        QoE += one_step_QoE

        if QoE < MIN_QOE:
            print(
                'Your QoE is too low...(Your video seems to have stuck forever) Please check for errors!')
            return np.array([-1e9, bandwidth_usage, QoE, sum_wasted_bytes, net_env.get_wasted_time_ratio()])

        if play_video_id >= ALL_VIDEO_NUM:
            print("The user leaves.")
            break

        start = timer()
        mem_before = get_process_memory()
        download_video_id, bit_rate, sleep_time = solution.run(
            delay, rebuf, video_size, end_of_video, play_video_id, net_env.players, False)
        mem_after = get_process_memory()
        end = timer()
        T_run.append(end - start)

        assert 0 <= download_video_id - play_video_id < len(net_env.players), "The video you choose is not in the current Recommend Queue. \
                \n   % You can only choose the current play video and its following four videos %"

    S = QoE - theta * bandwidth_usage * 8 / 1000000.
    print("Your score is: ", S)
    print("Your QoE is: ", QoE)
    print("Your sum of wasted bytes is: ", sum_wasted_bytes)
    W.append(sum_wasted_bytes * 8 / 1000000)
    if len(W) == 1000:
        print("my avg waste: ", np.sum(W) / len(W))
        print("my full waste: ", W)
    print("Your download/watch ratio (downloaded time / total watch time) is: ",
          net_env.get_wasted_time_ratio())

    Time_run.append(np.sum(T_run) / len(T_run))
    if len(Time_run) == 1000:
        print('Time: ', Time_run)

    return np.array([S, bandwidth_usage,  QoE, sum_wasted_bytes, net_env.get_wasted_time_ratio()])


SAMPLE_COUNT = 5
if __name__ == '__main__':
    assert args.trace in ["mixed", "high", "low", "medium"]

    if args.quickstart != '':
        isBaseline = False
        isQuickstart = True
        startTime = time.time()
        trace_id, user_sample_id = 1, 1

        solution = initialize_algorithm(isBaseline, isQuickstart, args.quickstart)
        net_env = initialize_environment(trace_id, user_sample_id)

        perform_action(solution,net_env, )
        
        print('Running time:', time.time() - startTime)
    else:
        print("Things gone wrong mate!")

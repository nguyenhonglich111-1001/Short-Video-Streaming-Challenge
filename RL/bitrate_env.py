# RL\bitrate_env.py

from constant.constants import BITRATE_LEVELS
import os
print("Current Working Directory:", os.getcwd())
import sys
sys.path.append(
    'C:\\Users\\LichNH\\Coding\\Short-Video-Streaming-Challenge')

from simulator import controller as env, short_video_load_trace
from timeit import default_timer as timer
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time
import os
import random
import numpy as np
import psutil


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

class BitrateEnv(gym.Env):
    def __init__(self):
        super(BitrateEnv, self).__init__()

        # Define action and observation space
        # Assuming 4 bitrate levels as actions
        self.action_space = spaces.Discrete(4)

        # Observation space can be a vector of relevant metrics
        # e.g., current buffer level, last chunk download time, etc.
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(3,), dtype=np.float32)

        # Initialize environment state
        self.state = np.zeros(3)
        self.buffer = 0
        self.last_chunk_time = 1
        self.total_chunks = 10
        self.current_chunk = 0

    def reset(self):
        self.state = np.zeros(3)
        self.buffer = np.random.uniform(0, 1)
        self.last_chunk_time = np.random.uniform(0.1, 1)
        self.current_chunk = 0

        # isBaseline = False
        # isQuickstart = True
        # startTime = time.time()
        # trace_type = "high"
        # trace_id, user_sample_id = 1, 1

        # load_trace(trace_type)
        # # reset the sample random seeds
        # seeds = np.random.randint(10000, size=(7, 2))


        return self.state

    def step(self, action):
        # Implement your MPC integration here
        bit_rate = mpc_module.mpc(past_bandwidth, past_bandwidth_ests, past_errors,
                       all_future_chunks_size, P, buffer_size,
                       chunk_sum, video_chunk_remain, last_quality,
                       Players, download_video_id, play_video_id,
                       future_bandwidth)

        # Simulate the download of a chunk with the selected bitrate
        download_time = MILLISECONDS_IN_SECOND * \
            (all_future_chunks_size[bit_rate]
             [self.current_chunk] / 1000000.0) / future_bandwidth

        # Update buffer
        self.buffer += self.last_chunk_time - download_time
        self.last_chunk_time = download_time

        # Update state
        self.state = np.array([self.buffer, self.last_chunk_time, bit_rate])

        # Calculate reward (assuming you have a method calculate_reward)
        reward = self.calculate_reward(bit_rate, download_time)

        # Check if episode is done
        done = self.current_chunk >= self.total_chunks
        self.current_chunk += 1

        return self.state, reward, done, {}

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


    def load_trace(trace_id):
        cooked_trace_folder = 'data/network_traces/' + str(trace_id) + '/'
        global all_cooked_time, all_cooked_bw
        all_cooked_time, all_cooked_bw = short_video_load_trace.load_trace(
            cooked_trace_folder)


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
            # mem_before = get_process_memory()
            download_video_id, bit_rate, sleep_time = solution.run(
                delay, rebuf, video_size, end_of_video, play_video_id, net_env.players, False)
            # mem_after = get_process_memory()
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


    def calculate_mpc_reward(self, bit_rate, download_time):
        # Implement reward calculation based on MPC module's reward formula
        bitrate_sum = ...  # Calculate bitrate sum based on selected bitrate
        rebuffer = ...  # Calculate rebuffer penalty based on download time
        smoothness_diffs = ...  # Calculate smoothness difference penalty
        waste = ...  # Calculate waste penalty (if applicable)

        # Use MPC hyperparameters from your module
        reward = (bitrate_sum * BITRATE_SUM_HYPER +
                  rebuffer * REBUFFER_HYPER +
                  smoothness_diffs * SMOOTH_DIFF_HYPER +
                  waste * WASTE_HYPER)

        return reward

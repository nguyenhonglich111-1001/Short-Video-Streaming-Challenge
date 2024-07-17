# RL\bitrate_env.py

import os
print("Current Working Directory:", os.getcwd())
import sys
# sys.path.append(
#     'C:\\Users\\LichNH\\Coding\\Short-Video-Streaming-Challenge')
sys.path.append(
    'D:\\Future_Internet_Lab\\Short-Video-Streaming-Challenge')
from constant.constants import BITRATE_LEVELS, VIDEO_BIT_RATE
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
        self.last_chunk_time = 1
        self.total_chunks = 10
        self.current_chunk = 0

        isBaseline = False
        isQuickstart = True
        trace_type = "high"
        trace_id, user_sample_id = 1, 1

        self.load_trace(trace_type)
        # reset the sample random seeds
        self.seeds = np.random.randint(10000, size=(7, 2))

        self.solution = self.initialize_algorithm(isBaseline, isQuickstart, "Inter_RL")
        self.net_env = self.initialize_environment(trace_id, user_sample_id)

        self.sum_wasted_bytes = 0
        self.QoE = 0
        self.T_run = []
        self.bandwidth_usage = 0
        self.quality = 0
        self.smooth = 0

        download_video_id, sleep_time = self.take_first_step(self.solution,self.net_env)
        self.download_video_id = download_video_id
        self.sleep_time = sleep_time

    

    def reset(self):
        self.state = np.zeros(3)
        self.last_chunk_time = np.random.uniform(0.1, 1)
        self.current_chunk = 0

        # isBaseline = False
        # isQuickstart = True
        # startTime = time.time()
        # trace_type = "high"
        # trace_id, user_sample_id = 1, 1

        # load_trace(trace_type)
        # # reset the sample random seeds
        # self.seeds = np.random.randint(10000, size=(7, 2))

        # self.sum_wasted_bytes = 0
        # self.QoE = 0
        # self.T_run = []
        # self.bandwidth_usage = 0
        # self.quality = 0
        # self.smooth = 0


        return self.state

    def get_process_memory():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss


    def get_smooth(self, net_env, download_video_id, chunk_id, quality):
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


    def load_trace(self, trace_id):
        cooked_trace_folder = 'data/network_traces/' + str(trace_id) + '/'
        global all_cooked_time, all_cooked_bw
        all_cooked_time, all_cooked_bw = short_video_load_trace.load_trace(
            cooked_trace_folder)


    def initialize_algorithm(self, isBaseline, isQuickstart, user_id):
        global LOG_FILE
        global log_file

        if isBaseline:
            if user_id == 'no_save':
                import baseline.no_save as Solution
                LOG_FILE = 'logs/log_nosave.txt'
                log_file = open(LOG_FILE, 'w')
        elif isQuickstart:
            sys.path.append('./quickstart/')
            if user_id == 'Inter_RL':
                import quickstart.Inter_RL as Solution
                LOG_FILE = 'logs/Inter_RL.txt'
                log_file = open(LOG_FILE, 'w')
            else:
                print("Not Imported My Dear Prof")
        else:
            sys.path.append(user_id)
            import solution as Solution
            sys.path.remove(user_id)
            LOG_FILE = 'logs/log.txt'
            log_file = open(LOG_FILE, 'w')

        solution = Solution.Algorithm()
        solution.Initialize()
        return solution


    def initialize_environment(self, trace_id, user_sample_id):
        net_env = env.Environment(
            user_sample_id, all_cooked_time[trace_id], all_cooked_bw[trace_id], ALL_VIDEO_NUM, seeds)
        return net_env


    def take_first_step(self, solution, net_env):
        download_video_id, sleep_time = solution.run(
            0, 0, 0, False, 0, net_env.players, True)
        assert 0 <= download_video_id <= 4, "The video you choose is not in the current Recommend Queue. \
            \n   % You can only choose the current play video and its following four videos %"
        return download_video_id, sleep_time

    def step(self, action):
        global last_chunk_bitrate
        global W
        global Time_run

        bit_rate = action

        if self.sleep_time == 0:
            max_watch_chunk_id = self.net_env.user_models[
                download_video_id - self.net_env.get_start_video_id()].get_watch_chunk_cnt()
            download_chunk = self.net_env.players[download_video_id -
                                                self.net_env.get_start_video_id()].get_chunk_counter()
            if max_watch_chunk_id >= download_chunk:
                if download_chunk == max_watch_chunk_id:
                    last_chunk_bitrate[download_video_id] = bit_rate
                    rel_id = download_video_id - self.net_env.get_start_video_id()
                    if rel_id + 1 < len(self.net_env.user_models):
                        if self.net_env.players[rel_id + 1].get_chunk_counter() != 0:
                            next_bitrate = self.net_env.players[rel_id + 1].get_downloaded_bitrate()[
                                0]
                            self.smooth += abs(self.quality -
                                            VIDEO_BIT_RATE[next_bitrate])
                self.quality = VIDEO_BIT_RATE[bit_rate]
                self.smooth += self.get_smooth(self.net_env, download_video_id,
                                            download_chunk, self.quality)

        delay, rebuf, video_size, end_of_video, play_video_id, waste_bytes = self.net_env.buffer_management(
            download_video_id, bit_rate, sleep_time)
        self.bandwidth_usage += video_size
        self.sum_wasted_bytes += waste_bytes

        one_step_QoE = alpha * self.quality / 1000. - beta * rebuf / 1000. - gamma * self.smooth / \
            1000.
        self.QoE += one_step_QoE

        if self.QoE < MIN_QOE:
            print(
                'Your self.QoE is too low...(Your video seems to have stuck forever) Please check for errors!')
            return np.array([-1e9, self.bandwidth_usage, self.QoE, self.sum_wasted_bytes, self.net_env.get_wasted_time_ratio()])

        # Update State
        self.state = np.array(
            [alpha * self.quality, beta * rebuf, gamma * self.smooth])

        if play_video_id >= ALL_VIDEO_NUM:
            print("The user leaves.")
            return self.state, one_step_QoE, True, {}

        start = timer()
        # mem_before = get_process_memory()
        download_video_id, sleep_time = self.solution.run(
            delay, rebuf, video_size, end_of_video, play_video_id, self.net_env.players, False)
        # mem_after = get_process_memory()
        
        end = timer()
        self.T_run.append(end - start)

        assert 0 <= download_video_id - play_video_id < len(self.net_env.players), "The video you choose is not in the current Recommend Queue. \
                \n   % You can only choose the current play video and its following four videos %"
        self.sleep_time = sleep_time
        if self.sleep_time != 0:
            pass
            # print("You choose to sleep for ", sleep_time, " ms", file=log_file)
        else:
            assert 0 <= download_video_id - play_video_id < len(self.net_env.players), "The video you choose is not in the current Recommend Queue. \
                \n   % You can only choose the current play video and its following four videos %"

        return self.state, one_step_QoE, False, {}
        # S = QoE - theta * self.bandwidth_usage * 8 / 1000000.
        # print("Your score is: ", S)
        # print("Your QoE is: ", QoE)
        # print("Your sum of wasted bytes is: ", self.sum_wasted_bytes)
        # W.append(self.sum_wasted_bytes * 8 / 1000000)
        # if len(W) == 1000:
        #     print("my avg waste: ", np.sum(W) / len(W))
        #     print("my full waste: ", W)
        # print("Your download/watch ratio (downloaded time / total watch time) is: ",
        #       self.net_env.get_wasted_time_ratio())

        # Time_run.append(np.sum(self.T_run) / len(self.T_run))
        # if len(Time_run) == 1000:
        #     print('Time: ', Time_run)

        # return np.array([S, bandwidth_usage,  QoE, sum_wasted_bytes, self.net_env.get_wasted_time_ratio()])

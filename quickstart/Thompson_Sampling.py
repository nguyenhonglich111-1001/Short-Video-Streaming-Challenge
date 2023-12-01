# Thompson Sampling-based Algorithm for Bitrate Selection in Video Streaming

# Initialize parameters for Thompson Sampling
# ...
import math
from simulator import mpc_module
from simulator.video_player import BITRATE_LEVELS
import numpy as np
import sys
sys.path.append("..")

MPC_FUTURE_CHUNK_COUNT = 5     # MPC
PAST_BW_LEN = 10
TAU = 500.0  # ms
PLAYER_NUM = 15
PROLOAD_SIZE = 800000.0   # B
PRELOAD_CHUNK_NUM = 4
RETENTION_THRESHOLD = 0.65
VIDEO_BIT_RATE = [750, 1200, 1850]  # Kilobit per second

# Parameters for Thompson Sampling
# BITRATE_LEVELS = 3  # Number of bitrate options
# Number of successes for each bitrate option
# Number of failures for each bitrate option

class Algorithm:
    def __init__(self):
        # Initialize any necessary parameters for the algorithm
        # Example: Initialize Thompson Sampling parameters, counters, or any other needed variables
        self.buffer_size = 0
        self.past_bandwidth = []
        self.past_bandwidth_ests = []
        self.past_errors = []
        self.sleep_time = 0
        self.future_bandwidth = 0.0
        self.avg_bandwidth = 0
        self.successes = np.zeros(BITRATE_LEVELS)
        self.failures = np.zeros(BITRATE_LEVELS)

    def estimate_bw(self):
        # record the newest error
        # curr_error = 0  # default assumes that this is the first request so error is 0 since we have never predicted bandwidth
        # if (len(self.past_bandwidth_ests) > 0) and self.past_bandwidth[-1] != 0:
        #     curr_error = abs(self.past_bandwidth_ests[-1] - self.past_bandwidth[-1])/float(self.past_bandwidth[-1])
        # self.past_errors.append(curr_error)
        # first get harmonic mean of last 5 bandwidths
        past_bandwidth = self.past_bandwidth[:]
        while past_bandwidth[0] == 0:
            past_bandwidth = past_bandwidth[1:]
        self.avg_bandwidth = sum(past_bandwidth)/len(past_bandwidth)
        future_bandw = 0
        for past_val in range(len(past_bandwidth)):
            if (past_val == 0):
                future_bandw = past_bandwidth[0]
            else:
                future_bandw = future_bandw*0.8 + past_bandwidth[past_val]*0.2
        self.future_bandwidth = future_bandw
        # future bandwidth prediction
        # divide by 1 + max of last 5 (or up to 5) errors
        # max_error = 0
        # error_pos = -5
        # if ( len(self.past_errors) < 5 ):
        #     error_pos = -len(self.past_errors)
        # max_error = float(max(self.past_errors[error_pos:]))
        # self.future_bandwidth = harmonic_bandwidth/(1+max_error)  # robustMPC here
        self.past_bandwidth_ests.append(self.future_bandwidth)
        # self.past_bandwidth = np.roll(self.past_bandwidth, -1)
        # self.past_bandwidth[-1] = future_bandwidth

    def Initialize(self):
        # Initialize the session or any necessary variables
        self.past_bandwidth = np.zeros(PAST_BW_LEN)

    def thompson_sampling(self):
        sampled_success_rates = np.random.beta(self.successes + 1, self.failures + 1)
        selected_bitrate = np.argmax(sampled_success_rates)
        return selected_bitrate

    def get_reward(self, bitrate):
        # Simulated reward calculation based on streaming performance for the selected bitrate
        # Example: Simulate reward based on bitrate's performance (replace this logic with actual reward calculation)
        # For illustration purposes, reward is randomly determined here

        threshold = 0.6  # Adjust this threshold based on your scenario
        reward = np.random.rand()  # Simulated reward: random value between 0 and 1

        # For demonstration, consider reward based on a threshold
        if reward >= threshold:
            # If reward surpasses the threshold, consider it a success
            self.successes[bitrate] += 1
            return 1  # Simulated success
        else:
            # If reward is below the threshold, consider it a failure
            self.failures[bitrate] += 1
            return 0  # Simulated failure

    def run(self, delay, rebuf, video_size, end_of_video, play_video_id, Players, first_step=False):
        Players[0].rebuf_time.append(rebuf)
        DEFAULT_QUALITY = 0
        if first_step:   # 第一步没有任何信息
            self.sleep_time = 0
            return 0, 0, self.sleep_time

        # download a chunk, record the bitrate and update the network
        if self.sleep_time == 0:
            self.past_bandwidth = np.roll(self.past_bandwidth, -1)
            self.past_bandwidth[-1] = (float(video_size) /
                                       1000000.0) / (float(delay) / 1000.0)  # MB / s
            # print(self.past_bandwidth)
        P = []
        all_future_chunks_size = []
        future_chunks_highest_size = []
        future_chunks_smallest_size = []
        for i in range(min(len(Players), PLAYER_NUM)):
            if Players[i].get_remain_video_num() == 0:      # download over
                P.append(0)
                all_future_chunks_size.append([0])
                future_chunks_highest_size.append([0])
                future_chunks_smallest_size.append([0])
                continue
            if i == 0:
                P.append(min(MPC_FUTURE_CHUNK_COUNT,
                         Players[i].get_remain_video_num()))
            else:
                P.append(min(2, Players[i].get_remain_video_num()))
            all_future_chunks_size.append(
                Players[i].get_undownloaded_video_size(P[-1]))
            future_chunks_highest_size.append(
                all_future_chunks_size[-1][BITRATE_LEVELS-1])
            future_chunks_smallest_size.append(all_future_chunks_size[-1][0])

        download_video_id = -1
        # get_chunk_counter: tổng chunk đã tải về
        # get_buffer_size
        # get_user_model: retention rate
        # get_remain_video_num: số chunk chưa tải
        # get_play_chunk: số chunk đã xem

        if play_video_id < 6:
            next_id = 1
            self.estimate_bw()
            for seq in range(1, min(len(Players), PLAYER_NUM)):
                start_chunk = int(Players[seq].get_play_chunk())
                _, user_retent_rate = Players[seq].get_user_model()
                cond_p = float(user_retent_rate[Players[seq].get_chunk_counter(
                )]) / float(user_retent_rate[start_chunk])
                b_max_next = max(
                    cond_p*((max(future_chunks_highest_size[seq])/1000000)/self.future_bandwidth), 3.5*math.e**(-0.3*self.future_bandwidth-0.15*seq))
                if (Players[next_id].get_buffer_size() > Players[seq].get_buffer_size()) and (Players[seq].get_buffer_size()/1000) <= b_max_next and Players[seq].get_remain_video_num() != 0:
                    next_id = seq

        # for seq in range(min(len(Players), PLAYER_NUM)):
            # calculate the possibility: P(user will watch the chunk which is going to be preloaded | user has watched from the beginning to the start_chunk)

            # print('user_retent_rate[start_chunk]: ',user_retent_rate)
            # update past_errors and past_bandwidth_ests

            # next_id = 0
            # if seq == 0 and len(Players) > 1 :
            #     for i in range(1, min(len(Players), PLAYER_NUM)):
            #         start_chunk = int(Players[i].get_play_chunk())
            #         _, user_retent_rate = Players[i].get_user_model()
            #         cond_p= float(user_retent_rate[Players[i].get_chunk_counter()]) / float(user_retent_rate[start_chunk])
            #         b_max_next=cond_p*((max(future_chunks_highest_size[i])/1000000)/self.future_bandwidth)
            #         if (Players[i].get_buffer_size()/1000)<=b_max_next and Players[i].get_remain_video_num() != 0:
            #             # next_id = i
            #             break
            start_chunk = int(Players[next_id].get_play_chunk())
            _, user_retent_rate = Players[next_id].get_user_model()
            cond_p = float(user_retent_rate[Players[next_id].get_chunk_counter(
            )]) / float(user_retent_rate[start_chunk])
            b_max_next = max(
                cond_p*((max(future_chunks_highest_size[next_id])/1000000)/self.future_bandwidth), 1+TAU/1000)

        start_chunk = int(Players[0].get_play_chunk())
        _, user_retent_rate = Players[0].get_user_model()
        cond_p = float(user_retent_rate[Players[0].get_chunk_counter(
        )]) / float(user_retent_rate[start_chunk])

        # if play_video_id < 6:
        #     b_max=cond_p*((max(future_chunks_highest_size[0])/1000000)/self.future_bandwidth)+b_max_next
        #     # print("threshold1: ")
        # else:
        # b_max=max(cond_p*((max(future_chunks_highest_size[0])/1000000)/self.future_bandwidth),1+TAU/1000)
        b_max = max(cond_p*((max(future_chunks_highest_size[0])/1000000) /
                    self.future_bandwidth), 3.5*math.e**(-0.3*self.future_bandwidth-0.15*0))
        # print("threshold2: ")
        if (Players[0].get_buffer_size()/1000) <= b_max and Players[0].get_remain_video_num() != 0:
            # if seq == 0 and len(Players) > 1 :
            #     print('upper: ',cond_p*((max(future_chunks_highest_size[seq])/1000000)/self.future_bandwidth)+b_max_next+1)
            # print("b_max",b_max)
            # print('buffer: ',Players[0].get_buffer_size()/1000)
            download_video_id = play_video_id

        if download_video_id == -1 and play_video_id < 6:
            if (Players[next_id].get_buffer_size()/1000) <= b_max_next and Players[next_id].get_remain_video_num() != 0:
                # print('buffer: ',Players[next_id].get_buffer_size()/1000)
                # print("b_max",b_max_next)
                download_video_id = play_video_id + next_id

        if download_video_id == -1:  # no need to download
            self.sleep_time = TAU
            # print("sleep")
            bit_rate = 0
            download_video_id = play_video_id
        else:
            download_video_seq = download_video_id - play_video_id
            buffer_size = Players[download_video_seq].get_buffer_size()  # ms
            video_chunk_remain = Players[download_video_seq].get_remain_video_num(
            )
            chunk_sum = Players[download_video_seq].get_chunk_sum()
            download_chunk_bitrate = Players[download_video_seq].get_downloaded_bitrate(
            )
            last_quality = DEFAULT_QUALITY
            if len(download_chunk_bitrate) > 0:
                last_quality = download_chunk_bitrate[-1]
            

            # bit_rate = mpc_module.mpc(self.past_bandwidth, self.past_bandwidth_ests, self.past_errors, all_future_chunks_size[download_video_seq], P[
            #                           download_video_seq], buffer_size, chunk_sum, video_chunk_remain, last_quality, Players, download_video_id, play_video_id, self.future_bandwidth)
            
            
            bit_rate = self.thompson_sampling()
            # # Simulate reward based on selected bitrate (get_reward function)
            # reward = self.get_reward(selected_bitrate)


            self.sleep_time = 0.0


        return download_video_id, bit_rate, self.sleep_time

a = Algorithm()
print(a.thompson_sampling())
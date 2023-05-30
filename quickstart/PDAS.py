# Comparison Algorithm: No saving approach
# No saving algorithm downloads the current playing video first.
# When the current playing video download ends, it preloads the videos in the following players periodically, with 800KB for each video.

import numpy as np
import sys
sys.path.append("..")
from simulator.video_player import BITRATE_LEVELS
from simulator import mpc_module
import math

MPC_FUTURE_CHUNK_COUNT = 5     # MPC 
PAST_BW_LEN = 5
TAU = 50.0  # ms
PLAYER_NUM = 5  
PROLOAD_SIZE = 800000.0   # B
PRELOAD_CHUNK_NUM = 4
RETENTION_THRESHOLD = 0.65

class Algorithm:
    def __init__(self):
        # fill your self params
        self.buffer_size = 0
        self.past_bandwidth = []
        self.past_bandwidth_ests = []
        self.past_errors = []
        self.sleep_time = 0
        self.future_bandwidth=0

    # Intial
    def Initialize(self):
        # Initialize your session or something
        # past bandwidth record
        self.past_bandwidth = np.zeros(PAST_BW_LEN)

    def estimate_bw(self, P):
        # record the newest error
        curr_error = 0  # default assumes that this is the first request so error is 0 since we have never predicted bandwidth
        if (len(self.past_bandwidth_ests) > 0) and self.past_bandwidth[-1] != 0:
            curr_error = abs(self.past_bandwidth_ests[-1] - self.past_bandwidth[-1])/float(self.past_bandwidth[-1])
        self.past_errors.append(curr_error)
        # first get harmonic mean of last 5 bandwidths
        past_bandwidth = self.past_bandwidth[:]
        while past_bandwidth[0] == 0.0:
            past_bandwidth = past_bandwidth[1:]
        bandwidth_sum = 0
        for past_val in past_bandwidth:
            bandwidth_sum += (1/float(past_val))
        harmonic_bandwidth = 1.0/(bandwidth_sum/len(past_bandwidth))

        # future bandwidth prediction
        # divide by 1 + max of last 5 (or up to 5) errors
        max_error = 0
        error_pos = -5
        if ( len(self.past_errors) < 5 ):
            error_pos = -len(self.past_errors)
        max_error = float(max(self.past_errors[error_pos:]))
        self.future_bandwidth = harmonic_bandwidth/(1+max_error)  # robustMPC here
        self.past_bandwidth_ests.append(harmonic_bandwidth)
        # self.past_bandwidth = np.roll(self.past_bandwidth, -1)
        # self.past_bandwidth[-1] = future_bandwidth

    # Define your algorithm
    def run(self, delay, rebuf, video_size, end_of_video, play_video_id, Players, first_step=False):
        Players[0].rebuf_time.append(rebuf)
        DEFAULT_QUALITY = 0
        if first_step:   # 第一步没有任何信息
            self.sleep_time = 0
            return 0, 0, self.sleep_time

        # download a chunk, record the bitrate and update the network 
        if self.sleep_time == 0:
            self.past_bandwidth = np.roll(self.past_bandwidth, -1)
            self.past_bandwidth[-1] = (float(video_size)/1000000.0) /(float(delay) / 1000.0)  # MB / s
            # print(self.past_bandwidth)
        P = []
        all_future_chunks_size = []
        future_chunks_highest_size = []

        for i in range(min(len(Players), PLAYER_NUM)):
            if Players[i].get_remain_video_num() == 0:      # download over
                P.append(0)
                all_future_chunks_size.append([0])
                future_chunks_highest_size.append([0])
                continue
            
            P.append(min(MPC_FUTURE_CHUNK_COUNT, Players[i].get_remain_video_num()))
            all_future_chunks_size.append(Players[i].get_undownloaded_video_size(P[-1]))
            future_chunks_highest_size.append(all_future_chunks_size[-1][BITRATE_LEVELS-1])



        download_video_id = -1
        # get_chunk_counter: tổng chunk đã tải về
        # get_buffer_size
        # get_user_model: retention rate
        # get_remain_video_num: số chunk chưa tải
        # get_play_chunk: số chunk đã xem
        for seq in range(0, min(len(Players), PLAYER_NUM)):
            # calculate the possibility: P(user will watch the chunk which is going to be preloaded | user has watched from the beginning to the start_chunk)  
            start_chunk = int(Players[seq].get_play_chunk())
            _, user_retent_rate = Players[seq].get_user_model()
            cond_p= float(user_retent_rate[Players[seq].get_chunk_counter()]) / float(user_retent_rate[start_chunk])
            # print('user_retent_rate[start_chunk]: ',user_retent_rate)
            # update past_errors and past_bandwidth_ests
            self.estimate_bw(P[seq])
            b_max=max(cond_p*((max(future_chunks_highest_size[seq])/1000000)/self.future_bandwidth),3.5*math.e**(-0.3*self.future_bandwidth-0.15*seq))
            if (Players[seq].get_buffer_size()/1000)<=b_max and Players[seq].get_remain_video_num() != 0:
                print('upper: ',cond_p*((max(future_chunks_highest_size[seq])/1000000)/self.future_bandwidth))
                print('lower: ',3.5*math.e**(-0.3*self.future_bandwidth-0.15*seq))
                print('buffer: ',Players[seq].get_buffer_size()/1000)
                download_video_id=play_video_id+seq
                break

        if download_video_id == -1:  # no need to download
            self.sleep_time = TAU
            bit_rate = 0
            download_video_id = play_video_id
        else:
            download_video_seq = download_video_id - play_video_id
            buffer_size = Players[download_video_seq].get_buffer_size()  # ms
            video_chunk_remain = Players[download_video_seq].get_remain_video_num()
            chunk_sum = Players[download_video_seq].get_chunk_sum()
            download_chunk_bitrate = Players[download_video_seq].get_downloaded_bitrate()
            last_quality = DEFAULT_QUALITY
            if len(download_chunk_bitrate) > 0:
                last_quality = download_chunk_bitrate[-1]
            # print("choosing bitrate for: ", download_video_id, ", chunk: ", Players[download_video_seq].get_chunk_counter())
            # print("past_bandwidths:", self.past_bandwidth[-5:], "past_ests:", self.past_bandwidth_ests[-5:])
            bit_rate = mpc_module.mpc(self.past_bandwidth, self.past_bandwidth_ests, self.past_errors, all_future_chunks_size[download_video_seq], P[download_video_seq], buffer_size, chunk_sum, video_chunk_remain, last_quality, Players, download_video_id, play_video_id)
            self.sleep_time = 0.0

        return download_video_id, bit_rate, self.sleep_time
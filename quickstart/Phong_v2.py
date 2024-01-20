# Comparison Algorithm: No saving approach
# No saving algorithm downloads the current playing video first.
# When the current playing video download ends, it preloads the videos in the following players periodically, with 800KB for each video.

import math
from simulator.video_player import Player
from simulator import mpc_module
from simulator.video_player import BITRATE_LEVELS
from numba import int64
from numba import float64
from numba import jit
import numpy as np
import sys
sys.path.append("..")

MPC_FUTURE_CHUNK_COUNT = 5     # MPC
PAST_BW_LEN = 20
TAU = 500.0  # ms
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
        self.future_bandwidth = 0.0
        self.avg_bandwidth = 0

        self.macd_upper_bound = 0.005
        self.macd_lower_bound = -0.005
        self.harmonic_samples = 20
    # Intial

    def Initialize(self):
        # Initialize your session or something
        # past bandwidth record
        # self.past_bandwidth = np.zeros(PAST_BW_LEN)
        pass
#

    @staticmethod
    @jit((float64[:], int64), nopython=True, nogil=True)
    def _ewma(arr_in, window):
        r"""Exponentialy weighted moving average specified by a decay ``window``
        to provide better adjustments for small windows via:

            y[t] = (x[t] + (1-a)*x[t-1] + (1-a)^2*x[t-2] + ... + (1-a)^n*x[t-n]) /
                (1 + (1-a) + (1-a)^2 + ... + (1-a)^n).

        Parameters
        ----------
        arr_in : np.ndarray, float64
            A single dimenisional numpy array
        window : int64
            The decay window, or 'span'

        Returns
        -------
        np.ndarray
            The EWMA vector, same length / shape as ``arr_in``

        Examples
        --------
        >>> import pandas as pd
        >>> a = np.arange(5, dtype=float)
        >>> exp = pd.DataFrame(a).ewm(span=10, adjust=True).mean()
        >>> np.array_equal(_ewma_infinite_hist(a, 10), exp.values.ravel())
        True
        """
        n = arr_in.shape[0]
        ewma = np.empty(n, dtype=float64)
        alpha = 2 / float(window + 1)
        w = 1
        ewma_old = arr_in[0]
        ewma[0] = ewma_old
        for i in range(1, n):
            w += (1-alpha)**i
            ewma_old = ewma_old*(1-alpha) + arr_in[i]
            ewma[i] = ewma_old / w
        return ewma

    def calc_harmonic_mean(self, data, samples):
        data = np.array(data)
        end_index = len(data)
        start_index = max(0, len(data) - samples)
        sum = 0.0

        for index in range(start_index, end_index):
            sum += 1/data[index]

        n = (end_index - start_index + 1)
        return n/sum

    def estimate_bw(self, P):

        # TODO
        # Estimate bandwidth of last segment -> self.past_bandwidth_ests
        # e bitrate of channel v for multimedia segment i ->
        #  the time consumed to download segment i ->
        # bandwidth and throughput ?

        past_bandwidths = np.array(self.past_bandwidth)

        self.avg_bandwidth = sum(past_bandwidths)/len(past_bandwidths)
        future_bandw = past_bandwidths[0]
        for bandwidth in past_bandwidths:
            future_bandw = future_bandw*0.8 + bandwidth*0.2

        self.future_bandwidth = future_bandw

        self.past_bandwidth_ests.append(self.future_bandwidth)

    def estimate_bw2(self, P):
        past_bandwidths: np.ndarray = np.array(self.past_bandwidth)

        self.avg_bandwidth = sum(past_bandwidths)/len(past_bandwidths)

        ema_short = self._ewma(past_bandwidths, 5)
        ema_long = self._ewma(past_bandwidths, 15)
        macd = ema_short[-1] - ema_long[-1]

        k = 21
        P_ZERO = 0.2
        last_bandwidth_est = self.past_bandwidth_ests[-1]
        last_bandwidth = past_bandwidths[-1]

        if self.macd_lower_bound <= macd <= self.macd_upper_bound:

            p_param = abs(last_bandwidth -
                          last_bandwidth_est)/last_bandwidth_est
            weight1 = 1 / (1 + np.exp(-k * (p_param - P_ZERO)))
            harmonic_mean = self.calc_harmonic_mean(
                past_bandwidths, self.harmonic_samples)

            future_bandw = weight1*harmonic_mean + \
                (1-weight1)*last_bandwidth
        else:
            past_bandwidth_avg = np.mean(past_bandwidths)
            triangle_param = (
                past_bandwidths[-1] - past_bandwidth_avg) / past_bandwidth_avg
            weight2 = 0.8
            weight2 = 1 / (1 + np.exp(k*triangle_param))
            
            future_bandw = weight2*last_bandwidth_est + \
                (1-weight2)*last_bandwidth

        self.future_bandwidth = future_bandw
        self.past_bandwidth_ests.append(self.future_bandwidth)

    # Define your algorithm
    def run(self, delay, rebuf, video_size, end_of_video, play_video_id, Players: list[Player], first_step=False):
        Players[0].rebuf_time.append(rebuf)
        DEFAULT_QUALITY = 0
        if first_step:   # 第一步没有任何信息
            self.sleep_time = 0
            return 0, 0, self.sleep_time

        # download a chunk, record the bitrate and update the network
        if self.sleep_time == 0:
            # self.past_bandwidth = np.roll(self.past_bandwidth, -1)
            # self.past_bandwidth[-1] = (float(video_size)/1000000.0) /(float(delay) / 1000.0)  # MB / s
            if len(self.past_bandwidth_ests) == 0:
                self.past_bandwidth_ests = [(
                    video_size/1000000.0) / (delay / 1000.0)]
            self.past_bandwidth.append((
                video_size/1000000.0) / (delay / 1000.0))   # MB / s
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
                P.append(min(5, Players[i].get_remain_video_num()))
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
        for seq in range(min(len(Players), PLAYER_NUM)):
            # calculate the possibility: P(user will watch the chunk which is going to be preloaded | user has watched from the beginning to the start_chunk)

            # update past_errors and past_bandwidth_ests
            # TODO Change back and forth between 2 version of estimate bw
            self.estimate_bw2(P[seq])
            # next_id = 0
            if seq == 0 and len(Players) > 1:
                for i in range(1, min(len(Players), PLAYER_NUM)):
                    start_chunk = int(Players[i].get_play_chunk())
                    _, user_retent_rate = Players[i].get_user_model()
                    cond_p = float(user_retent_rate[Players[i].get_chunk_counter(
                    )]) / float(user_retent_rate[start_chunk])
                    b_max_next = cond_p * \
                        ((max(
                            future_chunks_highest_size[i])/1000000)/self.future_bandwidth)
                    # b_max_next=max(cond_p*((max(future_chunks_highest_size[i])/1000000)/self.future_bandwidth),3.5*math.e**(-0.3*self.future_bandwidth-0.15*i))
                    if (Players[i].get_buffer_size()/1000) <= b_max_next and Players[i].get_remain_video_num() != 0:
                        # next_id = i
                        break
            start_chunk = int(Players[seq].get_play_chunk())
            _, user_retent_rate = Players[seq].get_user_model()
            cond_p = float(user_retent_rate[Players[seq].get_chunk_counter(
            )]) / float(user_retent_rate[start_chunk])
            # b_max=max(cond_p*((max(future_chunks_highest_size[seq])/1000000)/self.future_bandwidth),3.5*math.e**(-0.3*self.future_bandwidth-0.15*seq))
            # if (min(future_chunks_smallest_size[seq])/1000000)/self.avg_bandwidth >= 1:
            #     b_max=cond_p*((max(future_chunks_highest_size[seq])/1000000)/self.future_bandwidth)
            #     # print("threshold1: ")
            # else:
            #     if seq == 0 and len(Players) > 1 :
            #         b_max=cond_p*((max(future_chunks_highest_size[seq])/1000000)/self.future_bandwidth)+b_max_next+1
            #     else:
            #         b_max=max(cond_p*((max(future_chunks_highest_size[seq])/1000000)/self.future_bandwidth),1+TAU/1000)
            #     # print("threshold2: ")

            if seq == 0 and len(Players) > 1 and (min(future_chunks_smallest_size[seq])/1000000)/self.avg_bandwidth < 1:
                b_max = cond_p * \
                    ((max(
                        future_chunks_highest_size[seq])/1000000)/self.future_bandwidth)+b_max_next+1
            else:
                b_max = cond_p * \
                    ((max(
                        future_chunks_highest_size[seq])/1000000)/self.future_bandwidth)

            # if b_max > 4:
            #     b_max =4
            # elif b_max < 1+TAU/1000:
            #     b_max = 1+TAU/1000

            b_max = min(b_max, 4)
            b_max = max(b_max, 1+TAU/1000)
            # print(b_max)
            if (Players[seq].get_buffer_size()/1000) <= b_max and Players[seq].get_remain_video_num() != 0:
                # if seq == 0 and len(Players) > 1 :
                #     print('upper: ',cond_p*((max(future_chunks_highest_size[seq])/1000000)/self.future_bandwidth)+b_max_next+1)
                # print(b_max)
                # print('buffer: ',Players[seq].get_buffer_size()/1000)
                download_video_id = play_video_id+seq
                break

        if download_video_id == -1:  # no need to download
            self.sleep_time = TAU
            # print("sleep")
            bit_rate = 0
            download_video_id = play_video_id
        else:

            # Get data for mpc
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
            # print("choosing bitrate for: ", download_video_id, ", chunk: ", Players[download_video_seq].get_chunk_counter())
            # print("past_bandwidths:", self.past_bandwidth[-5:], "past_ests:", self.past_bandwidth_ests[-5:])
            # if((max(future_chunks_highest_size[seq])/1000000)/self.avg_bandwidth) < 0.8:
            #     bit_rate = 2
            # else:
            bit_rate = mpc_module.mpc(self.past_bandwidth, self.past_bandwidth_ests, self.past_errors, all_future_chunks_size[download_video_seq], P[
                                      download_video_seq], buffer_size, chunk_sum, video_chunk_remain, last_quality, Players, download_video_id, play_video_id, self.future_bandwidth)
            # bit_rate = 0

            self.sleep_time = 0.0

        return download_video_id, bit_rate, self.sleep_time

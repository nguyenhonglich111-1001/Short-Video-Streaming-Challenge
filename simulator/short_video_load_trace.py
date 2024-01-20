import os
import matplotlib.pyplot as plt
from testing.macd_test import _ewma
import numpy as np
COOKED_TRACE_FOLDER = './data/network_traces/middle/'
BW_ADJUST_PARA = 1


def load_trace(cooked_trace_folder=COOKED_TRACE_FOLDER):
    cooked_files = os.listdir(cooked_trace_folder)
    cooked_files.sort(key=lambda x: int(x))
    all_cooked_time = []
    all_cooked_bw = []
    for cooked_file in cooked_files:
        file_path = cooked_trace_folder + cooked_file
        cooked_time = []
        cooked_bw = []
        with open(file_path, 'rb') as f:
            for line in f:
                parse = line.split()
                cooked_time.append(float(parse[0]))
                cooked_bw.append(float(parse[1])*BW_ADJUST_PARA)
        all_cooked_time.append(cooked_time)
        all_cooked_bw.append(cooked_bw)

        # short_ema_np = _ewma(
        #     np.array(cooked_bw), 5)
        # long_ema_np = _ewma(
        #     np.array(cooked_bw), 15)
        # macd = short_ema_np - long_ema_np
        # # Create a plot
        # plt.figure(figsize=(16, 10))
        # plt.plot(cooked_time, macd,
        #          marker='o', linestyle='-', color='b')

        # # Set labels and title
        # plt.xlabel('Time')
        # plt.ylabel('Bandwidth')
        # plt.title('Bandwidth over Time')

        # # Show grid
        # plt.grid(True)

        # # Show plot
        # plt.show()



    return all_cooked_time, all_cooked_bw

# This example aims at helping you to learn what parameters you need to decide in your algorithm.
# It only gives you clues to things you can do to improve the algorithm, so it isn't necessarily reasonable.
# You need to find a better solution to balance QoE and bandwidth waste.
# You can run this example and get results by command: python run.py --quickstart fix_preload

# Description of fixed-preload algorithm
# Fixed-preload algorithm downloads the current playing video first.
# When the current playing video download ends, it preloads the videos in the recommendation queue in order.
# The maximum of preloading size is 4 chunks for each video.
# For each preloading chunk, if possibility (using data in user_ret to esimate) > RETENTION_THRESHOLD, it is assumed that user will watch this chunk so that it should be preloaded.
# It stops when all downloads end.

# We use buffer size to decide bitrate, here is the threshold.
LOW_BITRATE_THRESHOLD = 1000
HIGH_BITRATE_THRESHOLD = 2000
# If there is no need to download, sleep for TAU time.
TAU = 500.0  # ms
# max length of PLAYER_NUM
PLAYER_NUM = 5
# user retention threshold
RETENTION_THRESHOLD = 0.65
# fixed preload chunk num
PRELOAD_CHUNK_NUM = 4

# threshold (ms)
B1 = 4000
B2 = 2000
K = 5
DUC_RETENTION_THRESHOLD = 0.5

class Algorithm:
    def __init__(self):
        # fill the self params
        self.thrp_sample = []
        self.smooth_thrp = 0.0
        pass

    def Initialize(self):
        # Initialize the session or something
        pass

    def get_smooth_thrp(self):
        if self.smooth_thrp == 0.0:
            self.smooth_thrp = self.thrp_sample[-1]
        else:
            self.smooth_thrp = self.smooth_thrp * 0.8 + self.thrp_sample[-1] * 0.2
            # if self.thrp_sample[-1] > self.thrp_sample[-2] * 0.8:
            #     self.smooth_thrp = self.smooth_thrp * 0.8 + self.thrp_sample[-1] * 0.2
            # else:
            #     self.smooth_thrp = self.thrp_sample[-1]
        return self.smooth_thrp

    # Define the algorithm here.
    # The args you can get are as follows:
    # 1. delay: the time cost of your last operation
    # 2. rebuf: the length of rebufferment
    # 3. video_size: the size of the last downloaded chunk
    # 4. end_of_video: if the last video was ended
    # 5. play_video_id: the id of the current video
    # 6. Players: the video data of a RECOMMEND QUEUE of 5 (see specific definitions in readme)
    # 7. first_step: is this your first step?
    def run(self, delay, rebuf, video_size, end_of_video, play_video_id, Players, first_step=False):
        # Here we didn't make use of delay & rebuf & video_size & end_of_video.
        # You can use them or get more information from Players to help you make better decisions.

        # If it is the first step, you have no information of past steps.
        # So we return specific download_video_id & bit_rate & sleep_time.
        Players[0].rebuf_time.append(rebuf)
        if first_step:
            self.sleep_time = 0
            return 0, 0, 0.0
        
        # decide download video id
        download_video_id = -1
        if Players[0].get_remain_video_num() != 0:  # prefetch next chunk of current video if B_cur < B
            download_video_id = play_video_id
            # print("buffer current video", play_video_id )
        
        
        if download_video_id == -1 and play_video_id <6:
            if Players[1].get_remain_video_num() != 0:
                download_video_id = play_video_id + 1
                # print("buffer next video", play_video_id + 1)   

        if download_video_id == -1:  # no need to download, sleep for TAU time
            sleep_time = TAU
            bit_rate = 0
            download_video_id = play_video_id  # the value of bit_rate and download_video_id doesn't matter
        else:
            bit_rate = 2
            sleep_time = 0.0

        return download_video_id, bit_rate, sleep_time


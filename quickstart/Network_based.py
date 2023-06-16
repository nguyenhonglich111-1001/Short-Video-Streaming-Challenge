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


DUC_RETENTION_THRESHOLD = 0.5

class Algorithm:
    def __init__(self):
        # fill the self params
        self.thrp_sample = []
        self.smooth_thrp = 0.0
        self.B1 = 2000
        self.B2 = 2000
        self.K = PLAYER_NUM
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
    
    def get_avg_thrp(self, N):
        avg_thrp = 0
        if len(self.thrp_sample) < N:
            N_real = len(self.thrp_sample)
        else:
            N_real = N
        for i in range(N_real):
            avg_thrp += self.thrp_sample[-1-i]/N_real
        return avg_thrp
        

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
            return 0, 2, 0.0
        
        # decide buffer size B
        thrp = self.get_avg_thrp(10)
        # print(thrp)
        # if thrp >  2.5*Players[0].get_video_size(2):
        #     B1=2000

        # elif thrp >  2*Players[0].get_video_size(2): 
        #     B1=3000

        # elif thrp >  1.5*Players[0].get_video_size(2):    
        #     B1=3000

        # else:    
        #     B1=4000

        # decide download video id
        download_video_id = -1
        if Players[0].get_remain_video_num() != 0:  # prefetch next chunk of current video if B_cur < B
            if thrp >  2.5*Players[0].get_video_size(2):
                self.B1=2000
                self.K = PLAYER_NUM
            elif thrp >  2*Players[0].get_video_size(2): 
                self.B1=3000
                self.K = PLAYER_NUM
            elif thrp >  1.5*Players[0].get_video_size(2):    
                self.B1=3000
                self.K=4
            else:    
                self.B1=4000
                self.K = PLAYER_NUM
            # print("B1: ",self.B1)
            if Players[0].get_buffer_size() < self.B1:
                    # print("B: ", self.B1)
                    download_video_id = play_video_id
                    # print("buffer current video",download_video_id)
        
        
        if download_video_id == -1 and play_video_id < 6:
            # preload videos in PLAYER_NUM one by one
            # print("Selecting next video to prefetch")
            # if Players[min(min(len(Players), PLAYER_NUM),self.K)-1].get_buffer_size()==self.B2 and self.B2<self.B1:
            #     self.B2+=1000
            a=1
            # for seq in range(1, min(min(len(Players), PLAYER_NUM),self.K)):
            #     print("len player: ", len(Players))
            #     if Players[seq].get_buffer_size() < self.B2:
            #         download_video_id = play_video_id + seq
            #         print("buffer next video",download_video_id)
            #         break

            for seq in range(1, min(min(len(Players), PLAYER_NUM),self.K)):
                if(Players[a].get_buffer_size()>Players[seq].get_buffer_size()):
                    a=seq
                    # print("next video: ", play_video_id + a)

            if thrp >  2.5*Players[a].get_video_size(2):
                self.B1=2000
                self.K = PLAYER_NUM
            elif thrp >  2*Players[a].get_video_size(2): 
                self.B1=3000
                self.K = PLAYER_NUM
            elif thrp >  1.5*Players[a].get_video_size(2):    
                self.B1=3000
                self.K = 4
            else:    
                self.B1=4000
                self.K = PLAYER_NUM
            if Players[a].get_remain_video_num() != 0:
                if Players[a].get_buffer_size() < self.B1:
                    # print("B: ", self.B1)
                    download_video_id = play_video_id + a
                    # print("buffer next video",download_video_id)        
            # while seq < min(min(len(Players), PLAYER_NUM),self.K):
            #     if Players[seq].get_buffer_size() < self.B2:
            #         download_video_id = play_video_id + seq
            #         print("buffer next video",download_video_id)
            #         break 
            #     elif seq == min(min(len(Players), PLAYER_NUM),self.K)-1 and self.B2<self.B1:
            #         self.B2=self.B2+1000
            #         print("B2: ",self.B2)
            #         seq=0   
            #         continue    
            #     else:
            #         seq+=1
                
                

        if download_video_id == -1:  # no need to download, sleep for TAU time
            sleep_time = TAU
            # print("sleep: ", TAU)
            bit_rate = 0
            download_video_id = play_video_id  # the value of bit_rate and download_video_id doesn't matter
        else:
            # seq = download_video_id - play_video_id
            bit_rate = 2
            
            ins_thrp = video_size / (delay * 0.001)
            if(ins_thrp !=0):
                self.thrp_sample.append(ins_thrp)
            # print(video_size, delay, ins_thrp, Players[seq].get_video_size(0))
            # bit_rate = 2
            # for bit_rate in [2,1,0]:
            #     if Players[seq].get_video_size(bit_rate) < self.get_avg_thrp(10):
            #         break
                       # for bit_rate in [2,1,0]:
            #     if Players[0].get_remain_video_num() != 0:
            #         if Players[0].get_buffer_size() < self.B1:
            #             if Players[0].get_video_size(bit_rate) < self.get_avg_thrp(10):
            #                 break
            #     if Players[a].get_remain_video_num() != 0:
            #         if Players[a].get_buffer_size() < self.B1:
            #             if Players[a].get_video_size(bit_rate) < self.get_avg_thrp(10):
            #                 break
            sleep_time = 0.0
        # print(Players[0].get_video_size(bit_rate))

        return download_video_id, bit_rate, sleep_time


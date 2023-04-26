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
DUC_RETENTION_THRESHOLD = 0.6

class Algorithm:
    def __init__(self):
        # fill the self params
        self.thrp_sample = []
        self.smooth_thrp = 0.0
        self.average_thrp = 0.0
        pass

    def Initialize(self):
        # Initialize the session or something
        pass
    def average(self):
        #if len(self.thrp_sample)<11:
        self.average_thrp=sum(self.thrp_sample)/len(self.thrp_sample)
        #else:
            #for i in range(len(self.thrp_sample)-10,len(self.thrp_sample)):
            #    self.average_thrp=self.average_thrp+self.thrp_sample[i]
            #self.average_thrp=self.average_thrp/11
                #self.average_thrp = (self.thrp_sample[len(self.thrp_sample)-10]+self.thrp_sample[len(self.thrp_sample)-9]+self.thrp_sample[len(self.thrp_sample)-8]+self.thrp_sample[len(self.thrp_sample)-7]+self.thrp_sample[len(self.thrp_sample)-6]+self.thrp_sample[len(self.thrp_sample)-5]+self.thrp_sample[len(self.thrp_sample)-4]+self.thrp_sample[len(self.thrp_sample)-3]+self.thrp_sample[len(self.thrp_sample)-2]+self.thrp_sample[len(self.thrp_sample)-1]+self.thrp_sample[len(self.thrp_sample)])/10
        return self.average_thrp
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
            _, user_retent_rate = Players[0].get_user_model()
            if float(user_retent_rate[Players[0].get_chunk_counter()]) > DUC_RETENTION_THRESHOLD:
                if Players[0].get_buffer_size() < B1:
                    download_video_id = play_video_id
            else:
                if Players[0].get_buffer_size() < B2:
                    download_video_id = play_video_id
        
        
        if download_video_id == -1:
            # preload videos in PLAYER_NUM one by one
            # print("Selecting next video to prefetch")
            for seq in range(1, min(len(Players), PLAYER_NUM)):
                if Players[seq].get_remain_video_num() != 0:      # preloading hasn't finished yet 
                    # print(seq, Players[seq].get_buffer_size())
                    if Players[seq].get_buffer_size() < B2+1000-seq:
                        download_video_id = play_video_id + seq
                        break
                    # _, user_retent_rate = Players[seq].get_user_model()
                    # if float(user_retent_rate[Players[seq].get_chunk_counter()]) > DUC_RETENTION_THRESHOLD:
                    #     if Players[seq].get_buffer_size() < B1:
                    #         download_video_id = play_video_id + seq
                    #         break
                    # else:
                    #     if Players[seq].get_buffer_size() < B2:
                    #         download_video_id = play_video_id + seq
                    #         break
                    

        if download_video_id == -1:  # no need to download, sleep for TAU time
            sleep_time = TAU
            bit_rate = 0
            download_video_id = play_video_id  # the value of bit_rate and download_video_id doesn't matter
        else:
            seq = download_video_id - play_video_id
            # decide bitrate according to buffer size
            # if Players[seq].get_buffer_size() > HIGH_BITRATE_THRESHOLD:
            #     bit_rate = 2
            # elif Players[seq].get_buffer_size() > LOW_BITRATE_THRESHOLD:
            #     bit_rate = 1
            # else:
            #     bit_rate = 0
            # decide bitate according to network throughput
            
            ins_thrp = video_size / (delay * 0.001)
            #print("#")
            #print(ins_thrp)
            if(ins_thrp !=0):
                self.thrp_sample.append(ins_thrp)
            smooth_thrp = self.get_smooth_thrp()
            # print(video_size, delay, ins_thrp, Players[seq].get_video_size(0))
            #print("##")
            #print( self.average())
            if self.average()<200000:           
                for bit_rate in [1,0]:
                    if Players[seq].get_video_size(bit_rate) < smooth_thrp:
                        break
                sleep_time = 0.0
                #if self.average()<100000:

            else:
                for bit_rate in [2]:
                    if Players[seq].get_video_size(bit_rate) < smooth_thrp:
                        break
                sleep_time = 0.0
            #print(download_video_id)
            #print(bit_rate)   
                
            
        return download_video_id, bit_rate, sleep_time
# # MPC for no_save
# # import numpy as np
# # import fixed_env as env
# # import load_trace
# # import matplotlib.pyplot as plt
# import itertools
# from video_player import VIDEO_CHUNCK_LEN

# VIDEO_BIT_RATE = [750,1200,1850]  # Kilobit per second
# BITS_IN_BYTE = 8
# REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
# SMOOTH_PENALTY = 1
# MILLISECONDS_IN_SECOND = 1000.0


# def mpc(past_bandwidth, past_bandwidth_ests, past_errors, all_future_chunks_size, P, buffer_size, chunk_sum, video_chunk_remain, last_quality):
#     # print("MPC:::", buffer_size, "\n")

#     CHUNK_COMBO_OPTIONS = []

#     # make chunk combination options ,
#     for combo in itertools.product([0,1,2], repeat=P):
#         CHUNK_COMBO_OPTIONS.append(combo)

#     # ================== MPC =========================
#     # shouldn't change the value of past_bandwidth_ests and past_errors in MPC
#     copy_past_bandwidth_ests = past_bandwidth_ests
#     # print("past bandwidth ests: ", copy_past_bandwidth_ests)
#     copy_past_errors = past_errors
#     # print("past_errs: ", copy_past_errors)
    
#     curr_error = 0 # defualt assumes that this is the first request so error is 0 since we have never predicted bandwidth
#     if ( len(copy_past_bandwidth_ests) > 0 ):
#         curr_error = abs(copy_past_bandwidth_ests[-1]-past_bandwidth[-1])/float(past_bandwidth[-1])
#     copy_past_errors.append(curr_error)

#     # pick bitrate according to MPC           
#     # first get harmonic mean of last 5 bandwidths
#     past_bandwidths = past_bandwidth[-5:]
#     while past_bandwidths[0] == 0.0:
#         past_bandwidths = past_bandwidths[1:]
#     #if ( len(state) < 5 ):
#     #    past_bandwidths = state[3,-len(state):]
#     #else:
#     #    past_bandwidths = state[3,-5:]
#     bandwidth_sum = 0
#     for past_val in past_bandwidths:
#         bandwidth_sum += (1/float(past_val))
#     harmonic_bandwidth = 1.0/(bandwidth_sum/len(past_bandwidths))
#     # print("harmonic_bandwidth:", harmonic_bandwidth)

#     # future bandwidth prediction
#     # divide by 1 + max of last 5 (or up to 5) errors
#     max_error = 0
#     error_pos = -5
#     if ( len(copy_past_errors) < 5 ):
#         error_pos = -len(copy_past_errors)
#     max_error = float(max(copy_past_errors[error_pos:]))
#     future_bandwidth = harmonic_bandwidth/(1 + max_error)  # robustMPC here
#     # print("future_bd:", future_bandwidth)
#     copy_past_bandwidth_ests.append(harmonic_bandwidth)

#     # future chunks length (try 4 if that many remaining)
#     # last_index = int(chunk_sum - video_chunk_remain)
    
#     # if ( chunk_sum - last_index < 5 ):
#         # future_chunk_length = chunk_sum - last_index

#     # all possible combinations of 5 chunk bitrates (9^5 options)
#     # iterate over list and for each, compute reward and store max reward combination
#     max_reward = float('-inf')
#     best_combo = ()
#     start_buffer = buffer_size
#     # print("start_buffer:", start_buffer)

#     #start = time.time()
#     for combo in CHUNK_COMBO_OPTIONS:
#         # combo = full_combo[0:future_chunk_length]
#         # calculate total rebuffer time for this combination (start with start_buffer and subtract
#         # each download time and add 2 seconds in that order)
#         curr_rebuffer_time = 0
#         curr_buffer = start_buffer  # ms
#         bitrate_sum = 0
#         smoothness_diffs = 0
#         # last_quality = int( bit_rate )
#         # print(combo)
#         for position in range(0, len(combo)):
#             chunk_quality = combo[position]
#             # print(len(all_future_chunks_size[0]))
#             # print(chunk_quality)
#             # print(position)
#             # index = last_index + position + 1 # e.g., if last chunk is 3, then first iter is 3+0+1=4
#             download_time = MILLISECONDS_IN_SECOND * (all_future_chunks_size[chunk_quality][position]/1000000.)/(future_bandwidth) # this is MB/MB/s --> seconds
#             # print("download time:", MILLISECONDS_IN_SECOND, "*",  (all_future_chunks_size[chunk_quality][position]/1000000.), "/", future_bandwidth)
#             if ( curr_buffer < download_time ):
#                 curr_rebuffer_time += (download_time - curr_buffer)
#                 curr_buffer = 0
#             else:
#                 curr_buffer -= download_time
#             curr_buffer += VIDEO_CHUNCK_LEN
#             bitrate_sum += VIDEO_BIT_RATE[chunk_quality]
#             smoothness_diffs += abs(VIDEO_BIT_RATE[chunk_quality] - VIDEO_BIT_RATE[last_quality])
#             # bitrate_sum += BITRATE_REWARD[chunk_quality]
#             # smoothness_diffs += abs(BITRATE_REWARD[chunk_quality] - BITRATE_REWARD[last_quality])
#             last_quality = chunk_quality
#         # compute reward for this combination (one reward per 5-chunk combo)
#         # bitrates are in Mbits/s, rebuffer in seconds, and smoothness_diffs in Mbits/s
        
#         reward = (bitrate_sum/1000.) - (REBUF_PENALTY*curr_rebuffer_time/1000.) - (smoothness_diffs/1000.)
#         # reward = bitrate_sum - (8*curr_rebuffer_time) - (smoothness_diffs)
#         if ( reward >= max_reward ):
#             if (best_combo != ()) and best_combo[0] < combo[0]:
#                 best_combo = combo
#             else:
#                 best_combo = combo
#             max_reward = reward
#             # send data to html side (first chunk of best combo)
#             send_data = 0 # no combo had reward better than -1000000 (ERROR) so send 0
#             if ( best_combo != () ): # some combo was good
#                 send_data = best_combo[0]

#     bit_rate = send_data
#     return bit_rate


# # MPC for PDAS
# # import numpy as np
# # import fixed_env as env
# # import load_trace
# # import matplotlib.pyplot as plt
# import itertools
# from video_player import VIDEO_CHUNCK_LEN
# import numpy as np
# import sys
# sys.path.append("..")
# import math

# VIDEO_BIT_RATE = [750,1200,1850]  # Kilobit per second
# BITS_IN_BYTE = 8
# REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
# SMOOTH_PENALTY = 1
# MILLISECONDS_IN_SECOND = 1000.0
# PLAYER_NUM = 5  

# def mpc(past_bandwidth, past_bandwidth_ests, past_errors, all_future_chunks_size, P, buffer_size, chunk_sum, video_chunk_remain, last_quality, Players, download_video_id,  play_video_id):
#     # print("MPC:::", buffer_size, "\n")

#     CHUNK_COMBO_OPTIONS = []

#     # make chunk combination options
#     for combo in itertools.product([0,1,2], repeat=P):
#         CHUNK_COMBO_OPTIONS.append(combo)

#     # ================== MPC =========================
#     # shouldn't change the value of past_bandwidth_ests and past_errors in MPC
#     copy_past_bandwidth_ests = past_bandwidth_ests
#     # print("past bandwidth ests: ", copy_past_bandwidth_ests)
#     copy_past_errors = past_errors
#     # print("past_errs: ", copy_past_errors)
    
#     curr_error = 0 # defualt assumes that this is the first request so error is 0 since we have never predicted bandwidth
#     if ( len(copy_past_bandwidth_ests) > 0 ):
#         curr_error = abs(copy_past_bandwidth_ests[-1]-past_bandwidth[-1])/float(past_bandwidth[-1])
#     copy_past_errors.append(curr_error)

#     # pick bitrate according to MPC           
#     # first get harmonic mean of last 5 bandwidths
#     past_bandwidths = past_bandwidth[-5:]
#     while past_bandwidths[0] == 0.0:
#         past_bandwidths = past_bandwidths[1:]
#     #if ( len(state) < 5 ):
#     #    past_bandwidths = state[3,-len(state):]
#     #else:
#     #    past_bandwidths = state[3,-5:]
#     bandwidth_sum = 0
#     for past_val in past_bandwidths:
#         bandwidth_sum += (1/float(past_val))
#     harmonic_bandwidth = 1.0/(bandwidth_sum/len(past_bandwidths))
#     # print("harmonic_bandwidth:", harmonic_bandwidth)

#     # future bandwidth prediction
#     # divide by 1 + max of last 5 (or up to 5) errors
#     max_error = 0
#     error_pos = -5
#     if ( len(copy_past_errors) < 5 ):
#         error_pos = -len(copy_past_errors)
#     max_error = float(max(copy_past_errors[error_pos:]))
#     future_bandwidth = harmonic_bandwidth/(1 + max_error)  # robustMPC here
#     # print("future_bd:", future_bandwidth)
#     copy_past_bandwidth_ests.append(harmonic_bandwidth)

#     # future chunks length (try 4 if that many remaining)
#     # last_index = int(chunk_sum - video_chunk_remain)
    
#     # if ( chunk_sum - last_index < 5 ):
#         # future_chunk_length = chunk_sum - last_index


#     # all possible combinations of 5 chunk bitrates (9^5 options)
#     # iterate over list and for each, compute reward and store max reward combination
#     max_reward = float('-inf')
#     best_combo = ()
#     start_buffer = buffer_size
#     # print("start_buffer:", start_buffer)

#     #start = time.time()
#     for combo in CHUNK_COMBO_OPTIONS:
#         # combo = full_combo[0:future_chunk_length]
#         # calculate total rebuffer time for this combination (start with start_buffer and subtract
#         # each download time and add 2 seconds in that order)
#         rebuffer = 0
#         curr_buffer = Players[0].get_buffer_size()
#         buffer_video_next = start_buffer
#         bitrate_sum = 0
#         smoothness_diffs = 0
#         current_play_chunk=Players[0].get_play_chunk()
#         cost_sum=0
#         # last_quality = int( bit_rate )
#         # print(combo)
#         for position in range(0, len(combo)):
#             chunk_quality = combo[position]
#             # print(len(all_future_chunks_size[0]))
#             # print(chunk_quality)
#             # print(position)
#             # index = last_index + position + 1 # e.g., if last chunk is 3, then first iter is 3+0+1=4
#             download_time = MILLISECONDS_IN_SECOND * (all_future_chunks_size[chunk_quality][position]/1000000.)/(future_bandwidth) # this is MB/MB/s --> seconds
#             #cost
#             cost_sum+=all_future_chunks_size[chunk_quality][position]
#             #the number of chunk that user will watch until download chunk finished
#             k=(all_future_chunks_size[chunk_quality][position]/1000000.)/(future_bandwidth)
#             # print("download time:", MILLISECONDS_IN_SECOND, "*",  (all_future_chunks_size[chunk_quality][position]/1000000.), "/", future_bandwidth)
            
#             # calculate the possibility: P(user will watch the chunk which is going to be preloaded | user has watched from the beginning to the start_chunk)  
#             start_chunk = int(Players[download_video_id - play_video_id].get_play_chunk())
#             _, user_retent_rate = Players[download_video_id - play_video_id].get_user_model()
#             if (download_video_id == play_video_id):
#                 cond_p = float(user_retent_rate[Players[download_video_id - play_video_id].get_chunk_counter()+position]) / float(user_retent_rate[int(current_play_chunk)])   
#             else:
#                 cond_p = float(user_retent_rate[Players[download_video_id - play_video_id].get_chunk_counter()+position]) / float(user_retent_rate[start_chunk])
#             bitrate_sum += cond_p*VIDEO_BIT_RATE[chunk_quality]
#             smoothness_diffs += cond_p*abs(VIDEO_BIT_RATE[chunk_quality] - VIDEO_BIT_RATE[last_quality])
#             last_quality = chunk_quality
#             p_leave =1
#             # possibility user scroll => cause rebuff on the next video
#             for i in range(len(Players)):
#                 start_chunk = int(Players[i].get_play_chunk())
#                 _, user_retent_rate = Players[i].get_user_model()
#                 if start_chunk+k > Players[i].get_chunk_sum():
#                     k = Players[i].get_chunk_sum()-start_chunk
#                 p_stay=float(user_retent_rate[int(start_chunk+k)]) / float(user_retent_rate[start_chunk]) 
#                 if i==0:
#                     if current_play_chunk+k > Players[i].get_chunk_sum():
#                         k = Players[i].get_chunk_sum()-current_play_chunk
#                     p_stay=float(user_retent_rate[int(current_play_chunk+k)]) / float(user_retent_rate[int(current_play_chunk)])
#                     rebuffer += p_leave*p_stay*max(download_time-curr_buffer,0)
#                 else:
#                     start_chunk = int(Players[i-1].get_play_chunk())
#                     _, user_retent_rate = Players[i-1].get_user_model()
#                     if i==1:
#                         if current_play_chunk+k > Players[i-1].get_chunk_sum():
#                             k = Players[i-1].get_chunk_sum()-current_play_chunk
#                         p_leave=p_leave*(1-(float(user_retent_rate[int(current_play_chunk+k)]) / float(user_retent_rate[int(current_play_chunk)])))
#                     else:
#                         if start_chunk+k > Players[i-1].get_chunk_sum():
#                             k = Players[i-1].get_chunk_sum()-start_chunk
#                         p_leave=p_leave*(1-(float(user_retent_rate[int(start_chunk+k)]) / float(user_retent_rate[start_chunk])))
#                     if i == (download_video_id - play_video_id):   
#                         rebuffer += p_leave*p_stay*max(download_time-buffer_video_next,0)
#                     else:
#                         rebuffer += p_leave*p_stay*max(download_time-Players[i].get_buffer_size(),0)
#             #buffer and played chunk change each step in future
#             if ( curr_buffer < download_time ):
#                 # rebuffer += cond_p*(download_time - curr_buffer)
#                 current_play_chunk+=curr_buffer/1000
#                 curr_buffer = 0
#             else:
#                 current_play_chunk+=k
#                 curr_buffer -= download_time
            
#             if (download_video_id == play_video_id):
#                 curr_buffer += VIDEO_CHUNCK_LEN
#             else:
#                 buffer_video_next += VIDEO_CHUNCK_LEN
#             # bitrate_sum += BITRATE_REWARD[chunk_quality]
#             # smoothness_diffs += abs(BITRATE_REWARD[chunk_quality] - BITRATE_REWARD[last_quality])


#         # compute reward for this combination (one reward per 5-chunk combo)
#         # bitrates are in Mbits/s, rebuffer in seconds, and smoothness_diffs in Mbits/s
        
#         reward = (bitrate_sum/1000.) - (1.85*rebuffer/1000.) - (smoothness_diffs/1000.) - 0.5*cost_sum*8/1000000.
#         # reward = bitrate_sum - (8*curr_rebuffer_time) - (smoothness_diffs)
#         if ( reward >= max_reward ):
#             if (best_combo != ()) and best_combo[0] < combo[0]:
#                 best_combo = combo
#             else:
#                 best_combo = combo
#             max_reward = reward
#             # send data to html side (first chunk of best combo)
#             send_data = 0 # no combo had reward better than -1000000 (ERROR) so send 0
#             if ( best_combo != () ): # some combo was good
#                 send_data = best_combo[0]

#     bit_rate = send_data
#     return bit_rate,max_reward

# MPC for phong
# import numpy as np
# import fixed_env as env
# import load_trace
# import matplotlib.pyplot as plt
import itertools
import numpy as np
import copy
import sys
from video_player import VIDEO_CHUNCK_LEN
from video_player import Player
from constant.constants import VIDEO_BIT_RATE, VIDEO_BIT_RATE_INDEX
sys.path.append("..")
import math

BITS_IN_BYTE = 8
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
MILLISECONDS_IN_SECOND = 1000.0
PLAYER_NUM = 5  
class State:

    def __init__(self, 
        bitrate_sum,
        rebuffer,
        smoothness_diffs,
        waste,
        cost_sum,
        combo,
        curr_buffer,
        buffer_video_next,
        current_play_chunk
    ) -> None:
        self.bitrate_sum  : float = bitrate_sum
        self.rebuffer : float = rebuffer
        self.smoothness_diffs : float = smoothness_diffs
        self.waste : float = waste
        self.cost_sum  : float = cost_sum
        self.combo : list[int] = combo[:]
        self.curr_buffer  = curr_buffer
        self.buffer_video_next  = buffer_video_next
        self.current_play_chunk  = current_play_chunk

    
    def make_a_copy(self,
                    ):
        new_state = State(self.bitrate_sum,
                          self.rebuffer,
                          self.smoothness_diffs,
                          self.waste,
                          self.cost_sum,
                          self.combo[:],
                          self.curr_buffer,
                          self.buffer_video_next,
                          self.current_play_chunk)

        return new_state

def mpc(past_bandwidth, past_bandwidth_ests, past_errors, all_future_chunks_size, P, buffer_size, chunk_sum, video_chunk_remain, last_quality, Players : list[Player], download_video_id,  play_video_id, future_bandwidth):
    # print("MPC:::", buffer_size, "\n")




    # ================== MPC =========================
    # shouldn't change the value of past_bandwidth_ests and past_errors in MPC
    # copy_past_bandwidth_ests = past_bandwidth_ests
    # print("past bandwidth ests: ", copy_past_bandwidth_ests)
    # copy_past_errors = past_errors
    # # print("past_errs: ", copy_past_errors)
    
    # curr_error = 0 # defualt assumes that this is the first request so error is 0 since we have never predicted bandwidth
    # if ( len(copy_past_bandwidth_ests) > 0 ):
    #     curr_error = abs(copy_past_bandwidth_ests[-1]-past_bandwidth[-1])/float(past_bandwidth[-1])
    # copy_past_errors.append(curr_error)

    # # pick bitrate according to MPC           
    # # first get harmonic mean of last 5 bandwidths
    # past_bandwidths = past_bandwidth[-5:]
    # while past_bandwidths[0] == 0.0:
    #     past_bandwidths = past_bandwidths[1:]
    # #if ( len(state) < 5 ):
    # #    past_bandwidths = state[3,-len(state):]
    # #else:
    # #    past_bandwidths = state[3,-5:]
    # bandwidth_sum = 0
    # for past_val in past_bandwidths:
    #     bandwidth_sum += (1/float(past_val))
    # harmonic_bandwidth = 1.0/(bandwidth_sum/len(past_bandwidths))
    # # print("harmonic_bandwidth:", harmonic_bandwidth)

    # # future bandwidth prediction
    # # divide by 1 + max of last 5 (or up to 5) errors
    # max_error = 0
    # error_pos = -5
    # if ( len(copy_past_errors) < 5 ):
    #     error_pos = -len(copy_past_errors)
    # max_error = float(max(copy_past_errors[error_pos:]))
    # future_bandwidth = copy_past_bandwidth_ests[-1]  # robustMPC here
    # print("future_bd:", future_bandwidth)
    # copy_past_bandwidth_ests.append(harmonic_bandwidth)

    # future chunks length (try 4 if that many remaining)
    # last_index = int(chunk_sum - video_chunk_remain)
    
    # if ( chunk_sum - last_index < 5 ):
        # future_chunk_length = chunk_sum - last_index


    # all possible combinations of 5 chunk bitrates (9^5 options)
    # iterate over list and for each, compute reward and store max reward combination
    max_reward = float('-inf')
    best_combo = ()
    start_buffer = buffer_size
    # print("start_buffer:", start_buffer)

    # make chunk combination options
    # TODO Change repeat=P when done testing with DP
    states : list[State]= []
    for i in range(3):
        states.append(
            State(
                0,
                0,
                0,
                0,
                0,
                [i],
                Players[0].get_buffer_size(),
                start_buffer,
                int(Players[0].get_play_chunk())
            )

        )
    send_data = 0 

    for p_index in range(P):
        new_states = []
        for state in states:

            # combo = full_combo[0:future_chunk_length]
            # calculate total rebuffer time for this combination (start with start_buffer and subtract
            # each download time and add 2 seconds in that order)

            # last_quality = int( bit_rate )
            # print(state.combo)

            position = len(state.combo) - 1
            chunk_quality = state.combo[position]

            # print(len(all_future_chunks_size[0]))
            # print(chunk_quality)
            # print(position)
            # index = last_index + position + 1 # e.g., if last chunk is 3, then first iter is 3+0+1=4
            download_time = MILLISECONDS_IN_SECOND * (all_future_chunks_size[chunk_quality][position]/1000000.)/(future_bandwidth) # this is MB/MB/s --> seconds
            #cost
            state.cost_sum+=all_future_chunks_size[chunk_quality][position]
            #the number of chunk that user will watch until download chunk finished
            k=int((all_future_chunks_size[chunk_quality][position]/1000000.)/(future_bandwidth))
            # print("download time:", MILLISECONDS_IN_SECOND, "*",  (all_future_chunks_size[chunk_quality][position]/1000000.), "/", future_bandwidth)
            
            # calculate the possibility: P(user will watch the chunk which is going to be preloaded | user has watched from the beginning to the start_chunk)  
            # start_chunk = int(Players[download_video_id - play_video_id].get_play_chunk())
            # _, user_retent_rate = Players[download_video_id - play_video_id].get_user_model()
            # if (download_video_id == play_video_id):
            #     cond_p = float(user_retent_rate[Players[download_video_id - play_video_id].get_chunk_counter()+position]) / float(user_retent_rate[state.current_play_chunk])   
            # else:
            #     cond_p = float(user_retent_rate[Players[download_video_id - play_video_id].get_chunk_counter()+position]) / float(user_retent_rate[start_chunk])
            state.bitrate_sum += VIDEO_BIT_RATE[chunk_quality] #cond_p* VIDEO_BIT_RATE[chunk_quality]
            state.smoothness_diffs += abs(VIDEO_BIT_RATE[chunk_quality] - VIDEO_BIT_RATE[last_quality]) #cond_p*abs(VIDEO_BIT_RATE[chunk_quality] - VIDEO_BIT_RATE[last_quality])
            last_quality = chunk_quality
            p_leave =1
            # possibility user scroll => cause rebuff on the next video
            for i in range(min(2,len(Players))):
                start_chunk = int(Players[i].get_play_chunk())
                _, user_retent_rate = Players[i].get_user_model()
                if start_chunk+k > Players[i].get_chunk_sum():
                    k = Players[i].get_chunk_sum()-start_chunk
                p_stay=float(user_retent_rate[start_chunk+k]) / float(user_retent_rate[start_chunk]) 
                if i==0:
                    if state.current_play_chunk+k > Players[i].get_chunk_sum():
                        k = Players[i].get_chunk_sum()-state.current_play_chunk
                    p_stay=float(user_retent_rate[state.current_play_chunk+k]) / float(user_retent_rate[state.current_play_chunk])
                    state.rebuffer += p_leave*p_stay*max(download_time-state.curr_buffer,0)
                else:
                    start_chunk = int(Players[i-1].get_play_chunk())
                    _, user_retent_rate = Players[i-1].get_user_model()
                    if state.current_play_chunk+k > Players[i-1].get_chunk_sum():
                        k = Players[i-1].get_chunk_sum()-state.current_play_chunk
                    p_leave=p_leave*(1-(float(user_retent_rate[state.current_play_chunk+k])  / float(user_retent_rate[state.current_play_chunk])))
                    if i == (download_video_id - play_video_id):   
                        state.rebuffer += p_leave*p_stay*max(download_time-state.buffer_video_next,0)
                    else:
                        state.rebuffer += p_leave*p_stay*max(download_time-Players[i].get_buffer_size(),0)
                    if download_video_id == play_video_id:
                        state.waste += p_leave*all_future_chunks_size[chunk_quality][position]
            #buffer and played chunk change each step in future
            if ( state.curr_buffer < download_time ):
                # state.rebuffer += cond_p*(download_time - state.curr_buffer)
                state.current_play_chunk+=int(state.curr_buffer/1000)
                state.curr_buffer = 0
            else:
                state.current_play_chunk+=k
                state.curr_buffer -= download_time
            
            if (download_video_id == play_video_id):
                state.curr_buffer += VIDEO_CHUNCK_LEN
            else:
                state.buffer_video_next += VIDEO_CHUNCK_LEN
            # state.bitrate_sum += BITRATE_REWARD[chunk_quality]
            # state.smoothness_diffs += abs(BITRATE_REWARD[chunk_quality] - BITRATE_REWARD[last_quality])
                
            # compute reward for this combination (one reward per 5-chunk state.combo)
            # bitrates are in Mbits/s, state.rebuffer in seconds, and state.smoothness_diffs in Mbits/s
            # reward = (state.bitrate_sum/1000.) - (1.85*state.rebuffer/1000.) - (state.smoothness_diffs/1000.)
            reward = (state.bitrate_sum/1000.) - (1.85*state.rebuffer/1000.) - (state.smoothness_diffs/1000.) - (state.waste*8/1000000.)
            # reward = (state.bitrate_sum/1000.) - (1.85*state.rebuffer/1000.) - (state.smoothness_diffs/1000.) - 0.5*state.cost_sum*8/1000000.
            # reward = state.bitrate_sum - (8*curr_rebuffer_time) - (state.smoothness_diffs)
            if ( reward >= max_reward and len(state.combo) >= P-2):

                # WHAT ???
                if (best_combo != ()) and best_combo[0] < state.combo[0]:
                    best_combo = state.combo
                else:
                    best_combo = state.combo
                max_reward = reward
                # send data to html side (first chunk of best state.combo)
                send_data = 0 # no state.combo had reward better than -1000000 (ERROR) so send 0
                if ( best_combo != () ): # some state.combo was good
                    send_data = best_combo[0]

            if p_index != P-1:
                for index in VIDEO_BIT_RATE_INDEX:
                    # next_state = copy.deepcopy(state)
                    next_state = state.make_a_copy()
                    next_state.combo.append(index)
                    new_states.append(next_state)

        states = new_states
        
    bit_rate = send_data

    return bit_rate
import itertools
import numpy as np
import copy
import sys
from simulator.video_player import VIDEO_CHUNCK_LEN
from simulator.video_player import Player
from constant.constants import VIDEO_BIT_RATE, VIDEO_BIT_RATE_INDEX, BITRATE_LEVELS
import constant.reward_hyper_params as reward_hyper_params

BITRATE_SUM_HYPER, REBUFFER_HYPER, SMOOTH_DIFF_HYPER, WASTE_HYPER = reward_hyper_params.read_reward_hyper_params()
sys.path.append("..")

BITS_IN_BYTE = 8
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
MILLISECONDS_IN_SECOND = 1000.0
PLAYER_NUM = 5  


def reimport_reward_hyper():
    global BITRATE_SUM_HYPER, REBUFFER_HYPER, SMOOTH_DIFF_HYPER, WASTE_HYPER
    BITRATE_SUM_HYPER, REBUFFER_HYPER, SMOOTH_DIFF_HYPER, WASTE_HYPER = reward_hyper_params.read_reward_hyper_params()
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
        current_play_chunk,
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
        # self.p_leave = p_leave

    
    def make_a_copy(self,
                    ):
        rebuffer = 0
        waste = 0
        new_state = State(self.bitrate_sum,
                          self.rebuffer,
                          self.smoothness_diffs,
                          self.waste,
                          self.cost_sum,
                          self.combo,
                          self.curr_buffer,
                          self.buffer_video_next,
                          self.current_play_chunk,
                          )

        return new_state

def mpc(past_bandwidth, past_bandwidth_ests, past_errors, all_future_chunks_size, P, buffer_size, chunk_sum, video_chunk_remain, last_quality, Players : list[Player], download_video_id,  play_video_id, future_bandwidth):

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

    # Init the state for future dp
    states : list[State]= []
    for i in range(BITRATE_LEVELS):
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
                int(Players[0].get_play_chunk()),
            )

        )
    send_data = 0 
    # print("In MPC",BITRATE_SUM_HYPER, REBUFFER_HYPER, SMOOTH_DIFF_HYPER, WASTE_HYPER)
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
            if position == 0:
                # pass
                state.smoothness_diffs += abs(VIDEO_BIT_RATE[chunk_quality] - VIDEO_BIT_RATE[last_quality]) 
            else:
                left_position = position - 1
                left_chunk_quality = state.combo[left_position]
                state.smoothness_diffs += abs(
                    VIDEO_BIT_RATE[chunk_quality] - VIDEO_BIT_RATE[left_chunk_quality])
            p_leave =1
            
            # possibility user scroll => cause rebuff on the next video
            if len(state.combo) >= P:
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
            
            # reward = (state.bitrate_sum/1000.) - (1.85*state.rebuffer/1000.) - (state.smoothness_diffs/1000.) - (state.waste*8/1000000.)
            reward = (state.bitrate_sum*BITRATE_SUM_HYPER) + (state.rebuffer*REBUFFER_HYPER) + (state.smoothness_diffs*SMOOTH_DIFF_HYPER) + (state.waste*WASTE_HYPER)
            reward_component = [state.bitrate_sum,
                                state.rebuffer,
                                state.smoothness_diffs,
                                state.waste]
            # if len(state.combo) >= P:
            #     print("Combo", state.combo, reward)
            #     print(reward_component)
            #     print('....')
            # if state.combo == [3, 3, 3, 3, 0]:
            #     print("Combo",state.combo, reward)
            #     print(reward_component)
            #     print('....')
            if ( reward >= max_reward and len(state.combo) >= P-1):

                # WHAT ???
                if (best_combo != ()) and best_combo[0] < state.combo[0]:
                    best_reward_component = reward_component
                    best_combo = state.combo
                else:
                    best_reward_component = reward_component
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
    # print("Best Combo", best_combo, max_reward)
    # print(best_reward_component)
    # print('....')
    return bit_rate
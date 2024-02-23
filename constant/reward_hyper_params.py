def read_reward_hyper_params():
    with open('./constant/reward_hyper_params.txt', 'r') as f:
        lines = f.readlines()

        bitrate_sum = float(lines[0].split('\n')[0])
        rebuffer = float(lines[1].split('\n')[0])
        smoothness_diffs = float(lines[2].split('\n')[0])
        waste = float(lines[3].split('\n')[0])
        print('Reward Hyper',bitrate_sum, rebuffer, smoothness_diffs, waste)
        return bitrate_sum, rebuffer, smoothness_diffs, waste

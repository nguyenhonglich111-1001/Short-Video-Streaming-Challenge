def read_network_params():
    with open('./constant/network_const.txt','r') as f:
        lines = f.readlines()


        EMA_LONG_WINDOW = int(float(lines[0].split('\n')[0]))
        EMA_SHORT_WINDOW = int(float(lines[1].split('\n')[0]))
        K = float(lines[2].split('\n')[0])
        P_ZERO = float(lines[3].split('\n')[0])
        return EMA_LONG_WINDOW,EMA_SHORT_WINDOW,K,P_ZERO
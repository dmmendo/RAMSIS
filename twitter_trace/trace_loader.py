def get_real_trace(filename):
    f = open(filename,'r')
    trace = []
    timestamps = []
    for line in f:
        timestamps.append(int(line.strip().split(" ")[0]))
        trace.append(float(line.strip().split(" ")[-1])/1000000)
    interval_size = timestamps[1] - timestamps[0]
    cur_idx = 0
    res_trace = []
    for i in range(max(timestamps)+1):
        res_trace.append(trace[int(i/interval_size)])
    return res_trace

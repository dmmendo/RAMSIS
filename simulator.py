import numpy as np
import time
import bisect
import copy
from sklearn.linear_model import LinearRegression
import scipy

def sim_worker(args):
    smdp,pi_obj,idx,distribution,eval_seconds = args
    success = False
    try:
        violation_rate, accuracy, accuracy_list, total_query = get_results(smdp,pi_obj,distribution=distribution,eval_seconds=eval_seconds)
        success = True
    except Exception as e:
        print(idx,'had an exception:',e)
    print(idx,'finished')
    if success:
        return violation_rate, accuracy, accuracy_list, total_query, pi_obj, idx
    else:
        return None
        
def calc_violation_rate(action_history, batch_size_history, tau_history, d_history):
    violated_batches = (np.array(tau_history) > np.array(d_history))
    violation_rate = np.sum(np.array(batch_size_history) * violated_batches) / np.sum(batch_size_history)
    #violation_rate = np.sum(violated_batches) / len(batch_size_history)
    return violation_rate

def calc_accuracy(smdp, action_history, batch_size_history, tau_history, d_history):
    accuracy_list = [batch_size_history[i]*smdp.model_accuracy[smdp.get_model_selected(action_history[i])] \
            for i in range(len(action_history)) if action_history[i] != 'none' and tau_history[i] <= d_history[i] and batch_size_history[i] > 0]
    accuracy = np.sum(accuracy_list)
    success_batches = (np.array(tau_history) <= np.array(d_history)) 
    if np.sum(np.array(batch_size_history)*success_batches) == 0:
        return 0.
    return accuracy / np.sum(np.array(batch_size_history)*success_batches), accuracy_list

def get_arrival_trace_summary(smdp,arrival_history):
    total_time = 0
    total_cycles = 0
    trace = []
    cur_arrival = 0
    for entry in arrival_history:
        cur_arrival += 1
        total_time += entry
        if int(total_time/smdp.cycle_time) > total_cycles:
            trace.append(cur_arrival)
            total_cycles += 1
            cur_arrival = 0
    num_intervals = int(len(trace)/len(smdp.rate_list))*len(smdp.rate_list)
    trace = np.array(trace)[:num_intervals].reshape(-1,len(smdp.rate_list))
    return np.mean(trace,axis=0)

def get_results(smdp,pi_obj,distribution='poisson',eval_seconds=5*60):
    action_history, batch_size_history, tau_history, d_history, arrival_history = run_sim(pi_obj,smdp,eval_time=eval_seconds * 1000 * 1000,distribution=distribution)
    violation_rate = calc_violation_rate(action_history, batch_size_history, tau_history, d_history)
    accuracy,accuracy_list = calc_accuracy(smdp, action_history, batch_size_history, tau_history, d_history)
    total_query = np.sum(batch_size_history)
    #print('max_batch:',np.max(batch_size_history))
    print(get_arrival_trace_summary(smdp,arrival_history))
    return violation_rate, accuracy, accuracy_list, total_query

class InfLatencySampler:
    def __init__(self,smdp):
        self.smdp = smdp
        self.model_inf_time_predictors = []
        for model_lat_list in smdp.model_inf_time:
            y = np.max(model_lat_list,axis=1)
            X = np.array([bsize for bsize in range(1,len(model_lat_list)+1)]).reshape(-1,1)
            predictor = LinearRegression().fit(X,y)
            self.model_inf_time_predictors.append(predictor)
    
    def sample(self,inst_action,num_served):
        if num_served == 0 or inst_action == 'none' or not self.smdp.is_model_selected(inst_action):
            tau = 0.
        elif num_served > self.smdp.model_inf_time.shape[1]:
            model_selection = self.smdp.get_model_selected(inst_action)
            tau = self.model_inf_time_predictors[model_selection].predict([[num_served]])[0]
        else:
            model_selection = self.smdp.get_model_selected(inst_action)
            tau = np.random.choice(self.smdp.model_inf_time[model_selection][num_served-1], \
                                   p=self.smdp.model_inf_time_probmass[model_selection][num_served-1])
        return tau

def sample_next_arrival(smdp,cur_time):
    tau = np.random.exponential(1/(smdp.get_current_rate(cur_time)))
    return tau

def gamma_sample_next_arrival(smdp,cur_time):
    mu = 1/(smdp.get_current_rate(cur_time))
    k = 0.015625
    scale = mu/k
    tau = scipy.stats.gamma.rvs(a=k, scale=scale)
    return tau
    
def uniform_sample_next_arrival(smdp,cur_time):
    mean_arrival_time = 1/(smdp.get_current_rate(cur_time))
    tau = np.random.uniform(low=0.0, high=2*mean_arrival_time)
    return tau

def constant_sample_next_arrival(smdp,cur_time):
    mean_arrival_time = 1/(smdp.get_current_rate(cur_time))
    tau = mean_arrival_time
    return tau

def erlang_sample_next_arrival(smdp,cur_time):
    mean_arrival_time = 1/(smdp.get_current_rate(cur_time))
    tau = scipy.stats.erlang.rvs(a=1,scale=mean_arrival_time)
    return tau
    
def qrate_estimation_countwindow(arrival_history,window_size):
    if len(arrival_history) == 0:
        return 1000.
    total_time = np.sum(arrival_history[-(window_size-1):])
    return window_size/total_time

def qrate_estimation_timewindow(arrival_history,last_arrival_time,cur_time,window_size):
    if len(arrival_history) == 0:
        return 1000.
    cur_window_size = cur_time - last_arrival_time
    i = 0
    assert cur_window_size >= 0
    while i < len(arrival_history) and cur_window_size + arrival_history[-(1+i)] < window_size:
        cur_window_size += arrival_history[-(1+i)]
        i += 1
    return i/window_size

def qrate_estimation_maxtimewindow(arrival_history,last_arrival_time,cur_time):
    if len(arrival_history) == 0:
        return 1000.
    cur_window_size = cur_time - last_arrival_time
    window_sizes = [100000*(i+1) for i in range(10)]
    max_window_size = window_sizes[-1]
    count_list = []
    i = 0
    cur_window_idx = 0
    assert cur_window_size >= 0
    while i < len(arrival_history) and cur_window_size + arrival_history[-(1+i)] < max_window_size:
        if cur_window_size + arrival_history[-(1+i)] >= window_sizes[cur_window_idx]:
            count_list.append(i/window_sizes[cur_window_idx])
            cur_window_idx += 1
        cur_window_size += arrival_history[-(1+i)]
        i += 1
    count_list.append(i/max_window_size)
    return np.max(count_list)

def run_sim(pi_obj,smdp,eval_time,distribution,seed=0):
    total_time = 0
    last_arrival_time = 0
    cur_resp_times = np.array([])
    cur_inst_finish_times = np.array([0. for i in range(smdp.num_inst)])
    future_arrival_times = []
    tau_list = []
    np.random.seed(seed)
    
    model_latency_sampler = InfLatencySampler(smdp)
    
    arrival_history = []
    action_history = []
    batch_size_history = []
    tau_history = []
    d_history = []
    while total_time < eval_time:
        assert len(cur_resp_times) < 10000, "queue size exploded!"
        inst_action_list = pi_obj.get_action(
            cur_resp_times,
            cur_inst_finish_times,
            #cur_rate=smdp.get_current_rate(total_time),
            #cur_rate=qrate_estimation_countwindow(arrival_history,window_size=10),
            cur_rate=qrate_estimation_timewindow(arrival_history,last_arrival_time,total_time,window_size=500000),
            #cur_rate=qrate_estimation_maxtimewindow(arrival_history,last_arrival_time,total_time),
        )
        total_cur_serve = []
        for inst_action, inst_select, cur_served in inst_action_list:
            if not isinstance(inst_action,int) and inst_action[0] == 0 and inst_action[-1] > 0:
                wait_time = inst_action[-1]
                if len(tau_list) == 0 or wait_time < tau_list[0]:
                    bisect.insort(tau_list, wait_time)
            else:
                num_served = len(cur_served)
                assert len(cur_resp_times) == 0 or inst_action != 'none'
                model_lat = 0.
                if num_served > 0:
                    #model_lat = sample_model_lat(smdp,inst_action,num_served)
                    model_lat = model_latency_sampler.sample(inst_action,num_served)
                    assert cur_inst_finish_times[inst_select] <= 0.
                    cur_inst_finish_times[inst_select] = model_lat
                    bisect.insort(tau_list, model_lat)
                    #action_history.append(inst_action)
                    #batch_size_history.append(num_served)
                    #tau_history.append(model_lat)
                    #d_history.append(np.min(np.array(cur_resp_times)[cur_served]) if num_served > 0 else float('inf'))
                    action_history += list([inst_action]*num_served)
                    batch_size_history += list([1]*num_served)
                    tau_history += list([model_lat]*num_served)
                    d_history += list(np.array(cur_resp_times)[cur_served].tolist())
                total_cur_serve += list(cur_served)
        assert len(total_cur_serve) <= len(cur_resp_times)
        assert len(future_arrival_times) <= 1
        if len(future_arrival_times) == 0:
            if distribution == "poisson":
                next_arrival_time = sample_next_arrival(smdp,total_time)
            elif distribution == "uniform":
                next_arrival_time = uniform_sample_next_arrival(smdp,total_time)
            elif distribution == "constant":
                next_arrival_time = constant_sample_next_arrival(smdp,total_time)
            elif distribution == "erlang":
                next_arrival_time = erlang_sample_next_arrival(smdp,total_time)
            elif distribution == "gamma":
                next_arrival_time = gamma_sample_next_arrival(smdp,total_time)
            else:
                assert False, "distribution not found!"
            future_arrival_times.append(next_arrival_time)
            arrival_history.append(next_arrival_time)
            
        if len(tau_list) == 0 or tau_list[0] >= future_arrival_times[0]:
            tau = future_arrival_times[0]
            next_resp_times = [smdp.SLO]
            last_arrival_time = tau + total_time
        elif tau_list[0] < future_arrival_times[0]:
            tau = tau_list[0]
            next_resp_times = []
        else:
            assert False, "this should not be possible"
            
        total_time += tau
        if len(total_cur_serve) > 0:
            remaining_mask = np.full((len(cur_resp_times)),True)
            remaining_mask[total_cur_serve] = False
            #cur_resp_times = (np.array(cur_resp_times)[remaining_mask]).tolist()
            cur_resp_times = np.array(cur_resp_times)[remaining_mask]
        #cur_resp_times = [entry - tau for entry in cur_resp_times] + next_resp_times
        cur_resp_times = np.concatenate([cur_resp_times-tau,next_resp_times])
        #cur_inst_finish_times = [np.maximum(0,entry-tau) for entry in cur_inst_finish_times]
        cur_inst_finish_times = np.maximum(0.,cur_inst_finish_times-tau)
        tau_list = [entry for entry in (np.array(tau_list) - tau) if entry > 0.]
        future_arrival_times = [entry for entry in (np.array(future_arrival_times) - tau) if entry > 0.]
        assert len(cur_resp_times) == 0 or np.all(cur_resp_times[:-1] <= cur_resp_times[1:]), "queue should always be sorted!"
    return action_history, batch_size_history, tau_history, d_history, arrival_history

import numpy as np
from distribution import *
from simulator import *
import scipy
import time

@jit(nopython=True)
def compute_p_arr(arrival_count,num_inst,i_p,j_p,k_p,k_slide):
    
    res_arr = np.zeros((num_inst))
    cache = np.zeros((num_inst))
    max_k = np.maximum(0,arrival_count*num_inst+num_inst - (num_inst))
    for x in range(num_inst):
        min_k = np.maximum(0,num_inst-x)
        cur_k_slide = np.ascontiguousarray(k_slide[min_k:max_k][::-1])
        min_b = num_inst-x
        max_b = np.minimum(min_b+len(cur_k_slide),arrival_count*num_inst)
        cache[x] = np.dot(j_p[min_b:max_b],cur_k_slide)
        
        #cur_rem = arrival_count*num_inst+num_inst-1 - x
        #tmp = 0.
        #for j in range(num_inst-x,arrival_count*num_inst):
        #    k_low_idx = cur_rem-j
        #    if k_low_idx < 0:
        #        break
        #    tmp += j_p[j] * k_slide[k_low_idx]
        #cache[x] = tmp 
    
    for cur_q in range(num_inst):
        res = 0.
        for i in range(num_inst-cur_q):
            cur_res = cache[i+cur_q]
            
            cur_rem = arrival_count*num_inst+num_inst-1-i-cur_q
            for j in range(arrival_count*num_inst,arrival_count*num_inst+num_inst-cur_q):
                k_low_idx = cur_rem-j
                if k_low_idx < 0:
                    break
                cur_res += j_p[j] * k_slide[k_low_idx]
            
            res += i_p[i] * cur_res
        res_arr[cur_q] = res
    return res_arr
    
def state_partial_pmf(rate,tau,num_inst,new_queue_size,resp_time,i_p,j_p,k_p,k_slide):
    if new_queue_size == 0:
        cur_mu = rate*tau
        cur_all_k = np.arange(num_inst*new_queue_size,num_inst*(new_queue_size+1))
        p = scipy.stats.poisson.pmf(k=cur_all_k,mu=cur_mu)
        scale = np.arange(1,num_inst+1)[::-1]
        res_arr = []
        for i in range(num_inst):
            res_arr.append(np.sum(p[:scale[i]]))
        res_arr = np.array(res_arr)/len(resp_time)
    else:
        res_arr = compute_p_arr(new_queue_size,num_inst,i_p,j_p,k_p,k_slide)
    return res_arr

@jit(nopython=True)
def compute_q_slide_list(max_new_queue_size,queue_size,num_inst,batch,q_p):
    max_arrival_count = max_new_queue_size - (queue_size - batch)
    q_slide_list = np.zeros((max_new_queue_size+1,num_inst))
    
    for arrival_count in range(max_arrival_count+1):
        if arrival_count < 0:
            continue
        for extra_count in range(num_inst):
            q_low_idx = np.maximum(0,arrival_count*num_inst - extra_count)
            q_slide_list[arrival_count+(queue_size - batch),extra_count] = np.sum(q_p[q_low_idx:arrival_count*num_inst + num_inst - extra_count])
    return q_slide_list

@jit(nopython=True)
def batch_compute_p_arr_list(queue_size,num_inst,batch,i_p,j_p,k_p,num_resp_times):
    res_arr = np.zeros((num_inst,num_resp_times))
    cache = np.zeros((((queue_size)*num_inst)+batch*num_inst,num_resp_times))
    for i in range(batch*num_inst): 
        tmp = (i_p[i]*j_p[batch*num_inst-i:(queue_size)*num_inst])
        cache[batch*num_inst:i+(queue_size)*num_inst] += tmp
        #for j in range(batch*num_inst-i,(queue_size)*num_inst):
        #    cache[i+j] += i_p[i]*j_p[j]
    
    #for x in range((queue_size)*num_inst):
    #    k_low_idx = np.maximum(0,(queue_size-1)*num_inst-x)
    #    for k in range(k_low_idx,(queue_size)*num_inst-x):
    #        extra_count = (x + k) % num_inst
    #        res_arr[extra_count] += cache[x]*k_p[k]
    
    for k_iter in range(num_inst):
        num_k = (queue_size-1)*num_inst+k_iter+1
        tmp = cache[:num_k]*(k_p[:num_k,:][::-1,:])
        res_arr[((queue_size-1)*num_inst+k_iter)% num_inst] += np.sum(tmp,axis=0)
        
        #for x in range(num_k):
        #    k = (queue_size-1)*num_inst-x+k_iter
        #    extra_count = (x + k) % num_inst
        #    res_arr[extra_count] += tmp[x]
        #    #assert x + k == (queue_size-1)*num_inst+k_iter
    return res_arr

class VarBatchSMDP:
    """
    M/G/1 Queue batch service with optional N batch delay, 
    derministic service time, same SLO for all queries, given arrival rate as queries/microsecond
    """
    def __init__(self,
                 rate,
                 SLO,
                 model_inf_time,
                 model_inf_time_probmass,
                 model_accuracy,
                 resp_time=None,
                 arrival_dist="poisson",
                 num_inst=1,
                 max_batch_size=32,
                 max_action_batch=None,
                 prob_thresh=1.,
                 fixed_interval_time=True,
                 num_interval = 100,
                 **kwargs):
        self.prob_thresh=prob_thresh
        self.model_inf_time = np.array(model_inf_time) # [model][batch_size][probability_mass_idx]
        if self.model_inf_time.shape[1] < max_batch_size+1:
            lat_predictor = InfLatencySampler(self)
            new_model_inf_time = []
            for i in range(self.model_inf_time.shape[0]):
                new_model_inf_time.append([])
                for j in range(self.model_inf_time.shape[1],max_batch_size+1,1):
                    lat = lat_predictor.sample(inst_action=(i+1,j+1),num_served=j+1)
                    new_model_inf_time[-1].append([lat])
            self.model_inf_time = np.concatenate([self.model_inf_time,np.array(new_model_inf_time)],axis=1)
            assert self.model_inf_time.shape[1] == max_batch_size+1
        self.model_inf_time_probmass = np.array(model_inf_time_probmass)
        self.model_inf_time_probmass = np.ones_like(self.model_inf_time)
        self.N = np.minimum(max_batch_size,self.model_inf_time.shape[1]) # max queue size
        self.K = len(model_inf_time)
        self.num_inst = num_inst
        self.rate = rate # queries per microsecond
        self.SLO = SLO # response latency SLO for all queries
        self.model_accuracy = model_accuracy
        self.max_action_batch = max_action_batch
        if resp_time is not None:
            self.resp_time = np.sort(np.array(resp_time))
        else:
            if not fixed_interval_time:
                #self.resp_time = np.sort(list(set(self.model_inf_time.flatten()[self.model_inf_time.flatten() <= self.SLO].tolist())))
                model_ind,batch_ind = np.unravel_index(np.argsort(self.model_inf_time[:,:,0], axis=None), self.model_inf_time[:,:,0].shape)
                prev_model = model_ind[0]
                prev_batch = batch_ind[0]
                self.resp_time = [ self.model_inf_time[prev_model,prev_batch,0] ]
                prev_lat = self.resp_time[-1]
                min_dist = 0
                for i in range(1,len(model_ind)):
                    cur_lat = self.model_inf_time[model_ind[i],batch_ind[i],0]
                    prev_lat = self.model_inf_time[model_ind[i-1],batch_ind[i-1],0]
                    if cur_lat > self.SLO:
                        break
                    if model_ind[i] != model_ind[i-1] and cur_lat not in self.resp_time:
                        self.resp_time.append(cur_lat)
                    if model_ind[i] != model_ind[i-1] and prev_lat not in self.resp_time:
                        self.resp_time.append(prev_lat)

                self.resp_time = np.array(self.resp_time)
                if self.SLO > self.resp_time[-1]:
                    self.resp_time = np.append(self.resp_time,self.SLO)
                #if self.resp_time[0] > float('-inf'):
                #    self.resp_time = np.insert(self.resp_time,0,float('-inf'))
                if self.resp_time[0] > 0.:
                    self.resp_time = np.insert(self.resp_time,0,0.)

                #for entry in np.arange(0.,self.SLO,self.SLO/100):
                #    if entry not in self.resp_time:
                #        self.resp_time = np.insert(self.resp_time,0,entry)
                self.resp_time = np.sort(np.array(self.resp_time))
            else:
                self.resp_time = []
                #self.resp_time += [(-self.SLO/num_interval)*(i+1) for i in range(int(num_interval))]
                self.resp_time += [(self.SLO/num_interval)*(i+1) for i in range(int(num_interval))]

                self.resp_time.append(0.)
                self.resp_time = np.array(np.sort(np.unique(self.resp_time)))
            
        self.prob_cache = {}
        self.subprob_cache = {}
        self.reward_cache = {}
        self.q_slide_cache = {}
        self.states_cache = None
        self.active_states_cache = None
        self.state_queue_sizes_cache = None
        self.prof_time = [0.,0.]
    
    def clear_cache(self):
        self.prob_cache = {}
        self.subprob_cache = {}
        self.reward_cache = {}
        self.q_slide_cache = {}
        self.states_cache = None
        self.active_states_cache = None
        self.state_queue_sizes_cache = None        
    
    def isEnd(self,state):
        return False
    
    def actions(self, state):
        queue_size,resp_time_idx = state
        if queue_size == 0:
            return [(0,0)]
        elif queue_size > self.N:
            return [self.get_default_action(state)]
        possible_actions = []
        
        possible_num_served = [queue_size]
        
        if self.max_action_batch == "all":
            possible_num_served = [i+1 for i in range(queue_size)]
        elif self.max_action_batch is not None:
            i = 0
            while int(2**i) < queue_size and int(2**i) <= self.max_action_batch:
                possible_num_served.append(int(2**i))
                i += 1
        
        for num_served in possible_num_served:
            possible_actions += [(i+1,num_served) for i in range(self.K) if self.model_inf_time[i][num_served-1][-1] <= self.resp_time[resp_time_idx]] 
        """
        size_arr = np.arange(1,queue_size+1).reshape((1,queue_size))
        deadline_mask = self.model_inf_time[:,:queue_size,-1] <= self.resp_time[resp_time_idx]
        throughput_est = size_arr/self.model_inf_time[:,:queue_size,-1]
        batch_size_idx = np.argmax(deadline_mask*throughput_est,axis=1)
        valid_batch_sizes = np.max(deadline_mask,axis=1)
        for i in range(len(valid_batch_sizes)):
            if valid_batch_sizes[i] and (i+1,batch_size_idx[i]+1) not in possible_actions:
                possible_actions.append((i+1,batch_size_idx[i]+1))
        """
        if self.get_default_action(state) not in possible_actions:
            possible_actions.append(self.get_default_action(state))
        #possible_actions = sorted(possible_actions,key=lambda x: x[1], reverse=True)
        #possible_actions = sorted(possible_actions,key=lambda x: x[0]) 
        assert len(possible_actions) > 0
        return possible_actions
    
    def get_default_action(self,state):
        queue_size,resp_time_idx = state
        if queue_size > 0:
            return (1,queue_size)
        else:
            return (0,0)
        
    def is_model_selected(self, action):
        if action == 'none':
            return False
        else:
            return action[0] > 0
    
    def get_model_selected(self, action):
        return action[0] - 1
    
    def reward(self, state, action):
        if self.is_model_selected(action) and self.get_queue_size(state) <= self.N and not self.is_SLO_violated(state,action):
            return (self.model_accuracy[self.get_model_selected(action)]/self.get_queue_size(state))*self.get_num_served(state,action)
            #return self.model_accuracy[self.get_model_selected(action)]
        elif self.is_model_selected(action):
            return 0.
            #return -0.5*self.get_num_served(state,action)/self.get_queue_size(state)
        else:
            return 0.
        
    def is_SLO_violated(self, state, action):
        if self.get_queue_size(state) == 0:
            return False
        num_served = self.get_num_served(state,action)
        required_earliest_resp_time = self.get_earliest_resp_time(state)
        return self.model_inf_time[self.get_model_selected(action),num_served-1][-1] > (required_earliest_resp_time)
    
    def probState(self, new_state, state, action,i_p_list=None,j_p_list=None,k_p_list=None,k_slide_list=None,norm_arr=None):
        queue_size, resp_time_idx = state
        new_queue_size, new_resp_time_idx = new_state
        if queue_size > 0 and self.get_num_served(state,action) >= queue_size:
            assert False
        elif queue_size > 0 and self.get_num_served(state,action) < queue_size:
            assert False
        elif queue_size == 0 and new_queue_size == 1 and new_resp_time_idx == len(self.resp_time)-1:
            return 1.
        else:
            return 0.
    
    def get_all_queue_sizes(self):
        if self.state_queue_sizes_cache is not None:
            return self.state_queue_sizes_cache
        queue_sizes = []
        for state in self.states():
            queue_sizes.append(self.get_queue_size(state))
        self.state_queue_sizes_cache = np.array(queue_sizes)
        return self.state_queue_sizes_cache
    
    def states(self):
        if self.states_cache is not None:
            return self.states_cache
        
        result = []
        for n in range(self.N+1):
            for resp_time_idx in range(len(self.resp_time)):
                result.append((n,resp_time_idx))
        result.append((self.N+1,0))
        self.states_cache = result
        return self.states_cache
 
    def active_states(self):
        if self.active_states_cache is not None:
            return self.active_states_cache
        
        self.active_states_cache = [entry for entry in self.states() if not self.isEnd(entry)]
        return self.active_states_cache

    def discount(self):
        return 1.
        #return 0.95
    
    def get_intervals(self,queue_size,new_resp_time_idx,action):
        assert self.is_model_selected(action)
        cur_model_selection = self.get_model_selected(action)
        tau = self.model_inf_time[cur_model_selection][queue_size-1][-1]
        lower_resp_time = self.resp_time[new_resp_time_idx]
        if new_resp_time_idx+1 < len(self.resp_time):
            upper_resp_time = self.resp_time[new_resp_time_idx+1]
        else:
            upper_resp_time = self.SLO
        upper_arrival = tau - self.SLO + upper_resp_time
        if new_resp_time_idx > 0:
            lower_arrival = tau - self.SLO + lower_resp_time
        else:
            lower_arrival = 0.
        first_interv = np.maximum(0.,lower_arrival)
        second_interv = np.maximum(0.,upper_arrival) - np.maximum(0.,lower_arrival)
        third_interv = np.maximum(0.,tau - np.maximum(0.,upper_arrival))
        return first_interv, second_interv, third_interv
    
    def get_batch_interval(self,new_resp_time_idx,state,action):
        assert self.is_model_selected(action)
        queue_size, resp_time_idx = state
        cur_model_selection = self.get_model_selected(action)
        tau = self.model_inf_time[cur_model_selection][self.get_num_served(state,action)-1][-1]

        if new_resp_time_idx > 0:
            arrival_low = np.minimum(np.maximum(0.,self.resp_time[new_resp_time_idx]+tau-self.resp_time[resp_time_idx]),self.SLO-self.resp_time[resp_time_idx])
        else:
            arrival_low = 0.
        if new_resp_time_idx+1 < len(self.resp_time):
            arrival_high = np.minimum(np.maximum(0.,self.resp_time[new_resp_time_idx+1]+tau-self.resp_time[resp_time_idx]),self.SLO-self.resp_time[resp_time_idx])
        else:
            arrival_high = self.SLO-self.resp_time[resp_time_idx]

        first_interv = arrival_low
        second_interv = arrival_high - arrival_low
        third_interv = self.SLO-self.resp_time[resp_time_idx] - arrival_high
        fourth_interv = tau
        return first_interv, second_interv, third_interv, fourth_interv

    
    def succProbReward_vectorized(self, state, action):
        queue_size, resp_time_idx = state
        if (state,action) in self.prob_cache:
            prob_list = self.prob_cache[(state,action)]
        else:
            if queue_size > 0 and self.get_num_served(state,action) >= queue_size:
                if (queue_size,action) in self.subprob_cache:
                    subprob_list = self.subprob_cache[(queue_size,action)]
                else:
                    i_list = []
                    i_count_list = []
                    j_list = []
                    j_count_list = []
                    k_list = []
                    k_count_list = []
                    max_count = self.N
                    for new_resp_time_idx in range(len(self.resp_time)):
                        first_interv, second_interv, third_interv = self.get_intervals(queue_size,new_resp_time_idx,action)
                        i_list.append(np.full((self.num_inst),self.rate*first_interv))
                        i_count_list.append(np.arange(self.num_inst))
                        j_list.append(np.full((max_count*self.num_inst+self.num_inst),self.rate*second_interv))
                        j_count_list.append(np.arange(max_count*self.num_inst+self.num_inst))
                        k_list.append(np.full((max_count*self.num_inst+self.num_inst),self.rate*third_interv))
                        k_count_list.append(np.arange(max_count*self.num_inst+self.num_inst))
                    i_p_list = scipy.stats.poisson.pmf(k=i_count_list,mu=i_list)
                    j_p_list = scipy.stats.poisson.pmf(k=j_count_list,mu=j_list)
                    k_p_list = scipy.stats.poisson.pmf(k=k_count_list,mu=k_list)
                    padded_k_p_list = np.pad(k_p_list, [(0,0),(self.num_inst-1,self.num_inst-1 )], 'constant', constant_values=(0.,0.))
                    k_slide_list = np.squeeze(np.sum(np.lib.stride_tricks.sliding_window_view(padded_k_p_list,(1,self.num_inst)),axis=-1))
                    assert self.is_model_selected(action)
                    cur_model_selection = self.get_model_selected(action)
                    tau = self.model_inf_time[cur_model_selection][queue_size-1][-1]
                    subprob_list = []
                    for i in range(len(self.states())-1):
                        new_state = self.states()[i]
                        new_queue_size, new_resp_time_idx = new_state
                        cur_prob = state_partial_pmf(self.rate,
                                                        tau,
                                                        self.num_inst,
                                                        new_queue_size,
                                                        self.resp_time,
                                                        i_p=i_p_list[new_resp_time_idx],
                                                        j_p=j_p_list[new_resp_time_idx],
                                                        k_p=k_p_list[new_resp_time_idx],
                                                        k_slide=k_slide_list[new_resp_time_idx])
                        subprob_list.append(cur_prob)
                    subprob_list = np.array(subprob_list)
                    subprob_list = scipy.sparse.csr_matrix(subprob_list)
                    self.subprob_cache[(queue_size,action)] = subprob_list
                if resp_time_idx+1 < len(self.resp_time):
                    prev_interv_upper = self.SLO-self.resp_time[resp_time_idx]
                    prev_interv_lower = self.SLO-self.resp_time[resp_time_idx+1]
                    #init_p = my_pmf(lam=self.rate,t_a=prev_interv_lower,t_b=prev_interv_upper,k=np.arange(self.num_inst) + (queue_size-1)*self.num_inst)
                    init_p = scipy.stats.poisson.pmf(k=np.arange(self.num_inst) + (queue_size-1)*self.num_inst,mu=prev_interv_upper*self.rate)
                else:
                    prev_interv = (self.SLO-self.resp_time[resp_time_idx])
                    init_p = scipy.stats.poisson.pmf(k=np.arange(self.num_inst) + (queue_size-1)*self.num_inst,mu=prev_interv*self.rate)
                #init_p = np.ones((self.num_inst))/self.num_inst
                if any(init_p == 0.):
                    init_p = init_p + np.nextafter(0, 1)
                init_p = init_p/np.sum(init_p)
                assert np.isclose(np.sum(init_p),1.)
                prob_list = subprob_list.dot(init_p)
            elif queue_size > 0 and self.get_num_served(state,action) < queue_size:
                init_p = scipy.stats.poisson.pmf(
                            k=np.arange((queue_size-1)*self.num_inst,queue_size*self.num_inst),
                            mu=self.rate*(self.SLO-self.resp_time[resp_time_idx]))
                norm = np.sum(init_p)
                max_count = self.N
                assert self.is_model_selected(action)
                cur_model_selection = self.get_model_selected(action)
                tau = self.model_inf_time[cur_model_selection][self.get_num_served(state,action)-1][-1]
                if (queue_size,action) not in self.q_slide_cache:
                    q_cur_all_k = np.arange((max_count+1 - (queue_size-self.get_num_served(state,action)))*self.num_inst + self.num_inst)
                    q_cur_mu = np.full((len(q_cur_all_k)),self.rate*tau)
                    q_p = scipy.stats.poisson.pmf(k=q_cur_all_k,mu=q_cur_mu)
                    q_slide_list = compute_q_slide_list(max_new_queue_size=max_count,
                                                  queue_size=queue_size,
                                                  num_inst=self.num_inst,
                                                  batch=self.get_num_served(state,action),
                                                  q_p=q_p)
                    q_slide_list = scipy.sparse.csr_matrix(q_slide_list)
                    self.q_slide_cache[(queue_size,action)] = q_slide_list
                else:
                    q_slide_list = self.q_slide_cache[(queue_size,action)]
                if norm > self.prob_thresh:
                    i_list = np.full((len(self.resp_time),self.get_num_served(state,action)*self.num_inst),0.)
                    i_count_list = np.tile(np.arange(self.get_num_served(state,action)*self.num_inst),(len(self.resp_time),1))
                    j_list = np.full((len(self.resp_time),queue_size*self.num_inst),0.)
                    j_count_list = np.tile(np.arange(queue_size*self.num_inst),(len(self.resp_time),1))
                    k_list = np.full((len(self.resp_time),queue_size*self.num_inst),0.)
                    k_count_list = np.tile(np.arange(queue_size*self.num_inst),(len(self.resp_time),1))

                    
                    num_nonzero = 0
                    start_idx = 0
                    #found_zero = False
                    for arrival_time_idx in range(len(self.resp_time)):
                        first_interv, second_interv, third_interv, fourth_interv = self.get_batch_interval(
                                                                                        new_resp_time_idx=arrival_time_idx,
                                                                                        state=state,
                                                                                        action=action)
                        if second_interv > 0:
                            num_nonzero += 1
                        if num_nonzero == 0:
                            start_idx += 1
                        #if num_nonzero > 0 and second_interv <= 0:
                        #    found_zero = True
                        #assert not found_zero or second_interv == 0
                        if second_interv <= 0 and num_nonzero > 0:
                            break
                        i_list[arrival_time_idx] = self.rate*first_interv
                        j_list[arrival_time_idx] = self.rate*second_interv
                        k_list[arrival_time_idx] = self.rate*third_interv  

                    i_p_list = np.zeros(i_count_list.shape)
                    j_p_list = np.zeros(j_count_list.shape)
                    k_p_list = np.zeros(k_count_list.shape)
                    start_t = time.time()
                    i_p_list[start_idx:start_idx+num_nonzero] = scipy.stats.poisson.pmf(
                        k=i_count_list[start_idx:start_idx+num_nonzero],
                        mu=i_list[start_idx:start_idx+num_nonzero])
                    j_p_list[start_idx:start_idx+num_nonzero] = scipy.stats.poisson.pmf(
                        k=j_count_list[start_idx:start_idx+num_nonzero],
                        mu=j_list[start_idx:start_idx+num_nonzero])
                    k_p_list[start_idx:start_idx+num_nonzero] = scipy.stats.poisson.pmf(
                        k=k_count_list[start_idx:start_idx+num_nonzero],
                        mu=k_list[start_idx:start_idx+num_nonzero])
                    self.prof_time[0] += time.time()-start_t
                    start_t = time.time()
                    subprob_list = batch_compute_p_arr_list(
                                    queue_size=queue_size,
                                    num_inst=self.num_inst,
                                    batch=self.get_num_served(state,action),
                                    i_p=i_p_list[start_idx:start_idx+num_nonzero].T,
                                    j_p=j_p_list[start_idx:start_idx+num_nonzero].T,
                                    k_p=k_p_list[start_idx:start_idx+num_nonzero].T,
                                    num_resp_times=num_nonzero)
                    self.prof_time[1] += time.time()-start_t
                    prob_list = np.zeros((max_count+1,len(self.resp_time)))
                    prob_list[:,start_idx:start_idx+num_nonzero] = q_slide_list.dot(subprob_list)
                    prob_list = prob_list.flatten()
                    prob_list = np.array(prob_list)/norm
                else:
                    if any(init_p == 0):
                        init_p = init_p + np.nextafter(0, 1)
                    init_p = init_p/np.sum(init_p)
                    #init_p = np.ones((self.num_inst))/self.num_inst
                    correct_idx = self.quantize_deadline(self.resp_time[resp_time_idx]-tau)
                    prob_list = np.zeros((max_count+1,len(self.resp_time)))
                    prob_list[:,correct_idx] = q_slide_list.dot(init_p)
                    prob_list = prob_list.flatten()
                    #prob_list = np.zeros(((max_count+1)*len(self.resp_time)))
            else:
                prob_list = []
                for i in range(len(self.states())-1):
                    new_state = self.states()[i]
                    cur_prob = self.probState(new_state,state,action)
                    prob_list.append(cur_prob)
                prob_list = np.array(prob_list)
            total_sum = np.sum(prob_list)
            
            assert not np.isnan(total_sum)
            try:
                assert total_sum >= 0 and (total_sum <= 1. or np.isclose(total_sum,1.)), total_sum
            except:
                print("problem total sum:",total_sum,state,action,norm)
                #assert False
            #assert np.isclose(1.,total_sum), str(total_sum) + " " + str(state) + " " + str(action)
            prob_list[len(self.resp_time)*2-1] += np.sum(prob_list[:len(self.resp_time)])
            prob_list[:len(self.resp_time)] = 0
            prob_list = np.append(prob_list,np.maximum(0.,1-total_sum))
            if np.sum(prob_list) > 0:
                prob_list = np.array(prob_list)/np.sum(prob_list)
            assert len(prob_list) == len(self.states())
            
            prob_list = scipy.sparse.csr_matrix(prob_list.squeeze())
            self.prob_cache[(state,action)] = prob_list
        
        if (state,action) in self.reward_cache:
            reward_list = self.reward_cache[(state,action)]
        else:
            r = self.reward(state,action)
            self.reward_cache[(state,action)] = np.array([r])
        
        result = (self.states(),self.prob_cache[(state,action)],self.reward_cache[(state,action)])
        return result
    
    def quantize_deadline(self,deadline):
        #res = np.argmax((self.resp_time <= deadline) * self.resp_time)
        res = np.maximum(np.searchsorted(self.resp_time,deadline,side="right")-1,0)
        return res
        
    def to_state(self,response_times):
        queue_size = len(response_times)
        if queue_size > 0:
            state = (queue_size,self.quantize_deadline(np.min(response_times)))
        else:
            #state = (queue_size,self.quantize_deadline(float('inf')))
            state = (queue_size,len(self.resp_time)-1)
        return state
    
    def get_num_served(self,state,action):
        queue_size, resp_time_idx = state
        if self.is_model_selected(action):
            return action[-1]
        else:
            return 0
    
    def get_queue_size(self,state):
        queue_size, resp_time_idx = state
        return queue_size
    
    def get_resp_times(self,state):
        queue_size, resp_time_idx = state
        return [self.resp_time[resp_time_idx] for i in range(queue_size)]
    
    def get_earliest_resp_time(self,state):
        queue_size, resp_time_idx = state
        return self.resp_time[resp_time_idx]
import numpy as np
from distribution import *
from simulator import *


class GeneralSMDP:
    def __init__(self,
                 rate_list,
                 SLO,
                 model_inf_time,
                 model_inf_time_probmass,
                 model_accuracy,
                 resp_time=None,
                 arrival_pmf=poisson_pmf,
                 num_inst=1,
                 cycle_time=1000000,
                 max_batch_size=32,
                ):
        self.model_inf_time = np.array(model_inf_time) # [model][batch_size][probability_mass_idx]
        if self.model_inf_time.shape[1] < max_batch_size:
            lat_predictor = InfLatencySampler(self)
            new_model_inf_time = []
            for i in range(self.model_inf_time.shape[0]):
                new_model_inf_time.append([])
                for j in range(self.model_inf_time.shape[1],max_batch_size,1):
                    lat = lat_predictor.sample(inst_action=i+1,num_served=j+1)
                    new_model_inf_time[-1].append([lat])
            self.model_inf_time = np.concatenate([self.model_inf_time,np.array(new_model_inf_time)],axis=1)
            assert self.model_inf_time.shape[1] == max_batch_size
        self.model_inf_time_probmass = np.array(model_inf_time_probmass)
        self.model_inf_time_probmass = np.ones_like(self.model_inf_time)
        self.N = self.model_inf_time.shape[1] # max queue size
        self.K = len(model_inf_time)
        self.rate_list = rate_list # queries per microsecond
        self.SLO = SLO # response latency SLO for all queries
        self.arrival_pmf = arrival_pmf
        self.model_accuracy = model_accuracy
        self.num_inst = num_inst
        self.total_num_actions = self.K + 1
        self.cycle_time = cycle_time
        self.resp_time=None #resp_time should not be used by general smdp
        """
        if resp_time is not None:
            self.resp_time = np.sort(np.array(resp_time))
        else:
            self.resp_time = np.sort(list(set(self.model_inf_time.flatten()[self.model_inf_time.flatten() <= self.SLO].tolist())))
            if self.SLO > self.resp_time[-1]:
                self.resp_time = np.append(self.resp_time,self.SLO)
            if self.resp_time[0] > float('-inf'):
                self.resp_time = np.insert(self.resp_time,0,float('-inf'))
        """
        self.states_cache = None
        self.active_states_cache = None
    
    def is_model_selected(self, action):
        if isinstance(action,int):
            return action > 0
        else:
            return action[0] > 0
    
    def get_model_selected(self, action):
        if isinstance(action,int):
            return action -1
        else:
            return action[0] - 1
    
    def get_current_rate(self,cur_time):
        cur_idx = int(cur_time / self.cycle_time) % len(self.rate_list)
        return self.rate_list[cur_idx]

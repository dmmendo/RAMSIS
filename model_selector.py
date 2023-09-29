import numpy as np
import scipy

class PlainInstPi:
    def __init__(self,smdp,pi,safety_V_dict=None,safety_dict=None,V_dict=None,**kwargs):
        self.pi = pi
        self.smdp = smdp
        self.safety_V_dict = safety_V_dict
        if safety_V_dict is not None and "tau_list" in safety_V_dict:
            self.tau_list = np.array(safety_V_dict["tau_list"])
        else:
            self.tau_list = None
        self.V_dict = V_dict
        self.safety_dict = safety_dict
        if safety_V_dict is not None and len(safety_V_dict) > 1:
            safety_V = []
            for state in self.smdp.states():
                safety_V.append(safety_V_dict[state])
            self.safety_V = np.array(safety_V)
        self.safety_Q_cache = {}
        if safety_V_dict is not None:
            self.max_safety = np.max(self.safety_V)
        
    def get_action(self,cur_resp_times):
        #DEFAULT_FAIL_ACTION = 1 #lowest latency model
        #DEFAULT_FAIL_ACTION = 'none'
        
        state = self.smdp.to_state(cur_resp_times)
        if state not in self.pi:
            #print('need to increase max batch size!',self.smdp.get_queue_size(state))
            pass
        if state not in self.pi or self.pi[state] == 'none':
            #print(state,'need to increase queue size!')
            action = self.smdp.get_default_action(state)
            num_served = self.smdp.get_num_served(state,action)
        else:
            action = self.pi[state]
            num_served = self.smdp.get_num_served(state,action)
            
        return action, num_served

class StaticInstPi:
    def __init__(self,smdp,static_action,batch_delay=False,**kwargs):
        self.smdp = smdp
        self.static_action = static_action
        self.batch_delay = batch_delay
        self.cur_arrival_rate = self.smdp.rate/self.smdp.num_inst
        self.set_max_batch_size()
    
    def set_max_batch_size(self,):
        batch_mode = "aggressive"
        if batch_mode == "conservative":
            self.set_conservative_max_batch_size()
        elif batch_mode == "aggressive":
            self.set_aggressive_max_batch_size()
        else:
            assert False, "batch mode not found!"
            
    def set_conservative_max_batch_size(self):    
        this_max_batch_size = self.smdp.N
        throughput = (np.arange(1,this_max_batch_size+1)/self.smdp.model_inf_time[:,:this_max_batch_size,-1])     
        largest_batch_sizes = np.zeros((len(self.smdp.model_inf_time)))
        for i in range(len(self.smdp.model_inf_time)):
            j = 0
            while throughput[i,j] >= self.cur_arrival_rate and self.smdp.model_inf_time[i,j,-1] <= self.smdp.SLO/2:
                j += 1
            largest_batch_sizes[i] = j
        if all(largest_batch_sizes == 0):
            self.max_batch_size = int(this_max_batch_size)
        else:
            best_idx = np.argmax(self.smdp.model_accuracy*(largest_batch_sizes > 0))
            self.max_batch_size = int(largest_batch_sizes[best_idx])
    
    def set_aggressive_max_batch_size(self):
        this_max_batch_size = self.smdp.N
        static_load_mask = (np.arange(1,this_max_batch_size+1)/self.smdp.model_inf_time[:,:this_max_batch_size,-1]) >= self.cur_arrival_rate
        static_SLO_mask = self.smdp.model_inf_time[:,:this_max_batch_size,-1] <= self.smdp.SLO/2
        largest_batch_sizes = np.max(static_load_mask*static_SLO_mask*np.arange(1,this_max_batch_size+1),axis=1)
        if all(largest_batch_sizes == 0):
            self.max_batch_size = this_max_batch_size
        else:
            best_idx = np.argmax(self.smdp.model_accuracy*(largest_batch_sizes > 0))
            self.max_batch_size = largest_batch_sizes[best_idx]
    
    def get_action(self,cur_resp_times):
        if len(cur_resp_times) == 0:
            return 0,0
        elif self.batch_delay \
            and len(cur_resp_times) < self.max_batch_size \
            and np.min(cur_resp_times) > self.smdp.model_inf_time[self.static_action-1][self.max_batch_size-1][-1]:
            return (0,np.min(cur_resp_times)-self.smdp.model_inf_time[self.static_action-1][self.max_batch_size-1][-1]), 0
        else:
            return self.static_action, np.minimum(len(cur_resp_times),self.max_batch_size)

class OnlineBaselineInstPi:
    def __init__(self,smdp,batch_delay=False,mode="aggressive",**kwargs):
        self.smdp = smdp
        self.batch_delay = batch_delay
        self.mode = mode
        self.cur_arrival_rate = self.smdp.rate/self.smdp.num_inst
        self.set_max_batch_size()
        assert self.mode in ["aggressive","conservative","no-load","no-SLO"]

    def set_max_batch_size(self,):
        batch_mode = "aggressive"
        if batch_mode == "conservative":
            self.set_conservative_max_batch_size()
        elif batch_mode == "aggressive":
            self.set_aggressive_max_batch_size()
        else:
            assert False, "batch mode not found!"
            
    def set_conservative_max_batch_size(self):    
        this_max_batch_size = self.smdp.N
        throughput = (np.arange(1,this_max_batch_size+1)/self.smdp.model_inf_time[:,:this_max_batch_size,-1])     
        largest_batch_sizes = np.zeros((len(self.smdp.model_inf_time)))
        for i in range(len(self.smdp.model_inf_time)):
            j = 0
            while throughput[i,j] >= self.cur_arrival_rate and self.smdp.model_inf_time[i,j,-1] <= self.smdp.SLO/2:
                j += 1
            largest_batch_sizes[i] = j
        if all(largest_batch_sizes == 0):
            self.max_batch_size = int(this_max_batch_size)
        else:
            best_idx = np.argmax(self.smdp.model_accuracy*(largest_batch_sizes > 0))
            self.max_batch_size = int(largest_batch_sizes[best_idx])
    
    def set_aggressive_max_batch_size(self):
        this_max_batch_size = self.smdp.N
        static_load_mask = (np.arange(1,this_max_batch_size+1)/self.smdp.model_inf_time[:,:this_max_batch_size,-1]) >= self.cur_arrival_rate
        static_SLO_mask = self.smdp.model_inf_time[:,:this_max_batch_size,-1] <= self.smdp.SLO/2
        largest_batch_sizes = np.max(static_load_mask*static_SLO_mask*np.arange(1,this_max_batch_size+1),axis=1)
        if all(largest_batch_sizes == 0):
            self.max_batch_size = this_max_batch_size
        else:
            best_idx = np.argmax(self.smdp.model_accuracy*(largest_batch_sizes > 0))
            self.max_batch_size = largest_batch_sizes[best_idx]
    
    def get_model_selection(self,cur_resp_times,num_served,):
        DEFAULT_FAIL_ACTION = 1 #lowest latency model
        
        earliest_resp_time = np.min(cur_resp_times)

        deadline_mask = self.smdp.model_inf_time[:,num_served-1,-1] <= earliest_resp_time
        SLO_mask = self.smdp.model_inf_time[:,num_served-1,-1] <= self.smdp.SLO/2
        load_mask = (num_served/self.smdp.model_inf_time[:,num_served-1,-1]) >= self.cur_arrival_rate
        if self.mode == "aggressive":
            cur_mask = deadline_mask
        elif self.mode == "conservative":
            cur_mask = deadline_mask*SLO_mask*load_mask
        elif self.mode == "no-load":
            cur_mask = deadline_mask*SLO_mask
        elif self.mode == "no-SLO":
            cur_mask = deadline_mask*load_mask
            
        if not any(cur_mask):
            return DEFAULT_FAIL_ACTION
        else:
            return int(np.argmax(self.smdp.model_accuracy*cur_mask) + 1)
    
    def get_action(self,cur_resp_times):
        if len(cur_resp_times) == 0:
            return 0,0
        
        num_served = np.minimum(len(cur_resp_times),self.max_batch_size)

        action = self.get_model_selection(cur_resp_times,num_served)
        
        if self.batch_delay \
            and len(cur_resp_times) < self.max_batch_size \
            and np.min(cur_resp_times) > self.smdp.model_inf_time[action-1][self.max_batch_size-1][-1]:
            return (0,np.min(cur_resp_times)-self.smdp.model_inf_time[action-1][self.max_batch_size-1][-1]), 0
        else:
            return action, num_served

class JellyfishInstPi:
    def __init__(self,smdp,batch_delay=False,**kwargs):
        self.smdp = smdp
        self.batch_delay = batch_delay
        self.cur_arrival_rate = self.smdp.rate/self.smdp.num_inst
        self.set_max_batch_size_and_model()
    
    def set_max_batch_size_and_model(self,):
        batch_mode = "aggressive"
        if batch_mode == "conservative":
            self.set_conservative_max_batch_size_and_model()
        elif batch_mode == "aggressive":
            self.set_aggressive_max_batch_size_and_model()
        else:
            assert False, "batch mode not found!"        
    
    def set_conservative_max_batch_size_and_model(self,):
        this_max_batch_size = self.smdp.N
        throughput = (np.arange(1,this_max_batch_size+1)/self.smdp.model_inf_time[:,:this_max_batch_size,-1])     
        largest_batch_sizes = np.zeros((len(self.smdp.model_inf_time)))
        for i in range(len(self.smdp.model_inf_time)):
            j = 0
            while throughput[i,j] >= 1.05*self.cur_arrival_rate and self.smdp.model_inf_time[i,j,-1] <= self.smdp.SLO/2:
                j += 1
            largest_batch_sizes[i] = j
        if all(largest_batch_sizes == 0):
            self.max_batch_size = int(this_max_batch_size)
            self.model_select = 1 #lowest latency model
        else:
            best_idx = np.argmax(self.smdp.model_accuracy*(largest_batch_sizes > 0))
            self.max_batch_size = int(largest_batch_sizes[best_idx])
            self.model_select = int(best_idx + 1)
   
    def set_aggressive_max_batch_size_and_model(self):
        this_max_batch_size = self.smdp.N
        static_load_mask = (np.arange(1,this_max_batch_size+1)/self.smdp.model_inf_time[:,:this_max_batch_size,-1]) >= 1.05*self.cur_arrival_rate
        static_SLO_mask = self.smdp.model_inf_time[:,:this_max_batch_size,-1] <= self.smdp.SLO/2
        largest_batch_sizes = np.max(static_load_mask*static_SLO_mask*np.arange(1,this_max_batch_size+1),axis=1)
        if all(largest_batch_sizes == 0):
            self.max_batch_size = this_max_batch_size
            self.model_select = 1
        else:
            best_idx = np.argmax(self.smdp.model_accuracy*(largest_batch_sizes > 0))
            self.max_batch_size = largest_batch_sizes[best_idx]
            self.model_select = int(best_idx + 1)

    def get_action(self,cur_resp_times):
        if len(cur_resp_times) == 0:
            return 0,0
        
        action = self.model_select
        
        
        num_served = np.minimum(len(cur_resp_times),self.max_batch_size)
        
        if self.batch_delay \
            and len(cur_resp_times) < self.max_batch_size \
            and np.min(cur_resp_times) > self.smdp.model_inf_time[action-1][self.max_batch_size-1][-1]:
            return (0,np.min(cur_resp_times)-self.smdp.model_inf_time[action-1][self.max_batch_size-1][-1]), 0
        else:
            return action, num_served

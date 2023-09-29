import numpy as np

class RandomBalancePi:
    def __init__(self,smdp,inst_pi_obj_list,inst_weight,**kwargs):
        self.smdp = smdp
        self.inst_pi_obj_list = inst_pi_obj_list
        self.inst_weight = inst_weight
        self.total_count = np.zeros((smdp.num_inst),dtype=np.int64)
        self.inst_queue = [[] for i in range(smdp.num_inst)]
        self.sched_call_count = 0
        self.batch_size_profile = []
        
    def synchronize_state(self,other_pi_obj):
        other_pi_obj.inst_queue = self.inst_queue
    
    def queue_new_queries(self,cur_resp_times,cur_inst_finish_times,inst_queue):
        if len(cur_resp_times) == 0:
            return
        cur_queue_size = np.sum([len(entry) for entry in inst_queue])     
        num_new_queue = len(cur_resp_times) - cur_queue_size
        inst_assignment = np.random.choice(self.smdp.num_inst,size=num_new_queue,p=self.inst_weight)
        for i in range(cur_queue_size,len(cur_resp_times),1):
            inst_select = inst_assignment[i-cur_queue_size]
            inst_queue[inst_select].append(i)
        assert np.sum([len(entry) for entry in inst_queue]) == len(cur_resp_times)
            
    def get_action_list(self,cur_resp_times,cur_inst_finish_times,inst_queue,allow_bad_states=False,random=False):
        action_list = []
        num_avail_to_serve = len(cur_resp_times)
        avail_inst = np.array(cur_inst_finish_times) <= 0
        inst_to_sched = np.nonzero(avail_inst)[0]
        #sched_order = np.argsort([inst_queue[inst_select][0] for inst_select in inst_to_sched if len(inst_queue[inst_select]) > 0])
        start_idx = 0
        total_serve = 0
        for inst_select in inst_to_sched:
            cur_queue = inst_queue[inst_select]
            try:
                inst_action, num_served = self.inst_pi_obj_list[inst_select].get_action(np.array(cur_resp_times)[cur_queue])
            except Exception as e:
                inst_action = 0
                num_served = 0
                if not allow_bad_states:
                    print(1/self.smdp.rate)
                    print(cur_resp_times)
                    print(len(cur_resp_times),cur_queue)
                    print(start_idx)
                    print(inst_select,cur_queue)
                    raise e
            cur_served = cur_queue[:num_served]
            start_idx += num_served
            total_serve += len(cur_served)
            action_list.append((inst_action, inst_select, cur_served))
        assert len(inst_to_sched) != 0 or start_idx == 0
        return action_list, total_serve
        
    def commit_action_list(self,cur_resp_times,cur_inst_finish_times,inst_queue,action_list):
        residual_q = 0
        total_serve = 0
        for inst_action, inst_select, cur_served in action_list:
            num_served = len(cur_served)
            self.batch_size_profile.append(num_served)
            residual_q += np.maximum(num_served - len(inst_queue[inst_select]),0)
            inst_queue[inst_select] = inst_queue[inst_select][num_served:]
            self.total_count[inst_select] += num_served
            total_serve += num_served
        
        total_remain = []
        for inst_select in range(len(inst_queue)):
            total_remain += list(inst_queue[inst_select])
        
        sorted_idx = np.sort(total_remain)
        new_idx_map = dict((sorted_idx[i],i) for i in range(len(sorted_idx)))
        for inst_select in range(len(inst_queue)):
            cur_queue = inst_queue[inst_select]
            for i in range(len(cur_queue)):
                cur_queue[i] = new_idx_map[cur_queue[i]]
        
        assert residual_q == 0
        assert np.sum([len(entry) for entry in inst_queue]) == len(cur_resp_times) - total_serve, "fail"
    
    def get_action(self,cur_resp_times,cur_inst_finish_times):
        if len(cur_resp_times) == 0 or all(np.array(cur_inst_finish_times) > 0.):
            return []
        self.sched_call_count += 1
        self.queue_new_queries(cur_resp_times,cur_inst_finish_times,self.inst_queue)
        action_list,total_query = self.get_action_list(cur_resp_times,cur_inst_finish_times,self.inst_queue)
        self.commit_action_list(cur_resp_times,cur_inst_finish_times,self.inst_queue,action_list)
        return action_list
    
class DeterministicRoundRobinPi:
    def __init__(self,smdp,inst_pi_obj_list,inst_weight,**kwargs):
        self.smdp = smdp
        self.inst_pi_obj_list = inst_pi_obj_list
        self.inst_weight = inst_weight
        self.total_count = np.zeros((smdp.num_inst),dtype=np.int64)
        self.inst_queue = [[] for i in range(smdp.num_inst)]
        self.sched_call_count = 0
        self.next_inst = 0
        self.batch_size_profile = []
    
    def synchronize_state(self,other_pi_obj):
        other_pi_obj.inst_queue = self.inst_queue
        other_pi_obj.next_inst = self.next_inst
    
    def queue_new_queries(self,cur_resp_times,cur_inst_finish_times,inst_queue):
        if len(cur_resp_times) == 0:
            return
        cur_queue_size = np.sum([len(entry) for entry in inst_queue])     
        num_new_queue = len(cur_resp_times) - cur_queue_size
        for i in range(cur_queue_size,len(cur_resp_times),1):
            inst_select = self.next_inst
            inst_queue[inst_select].append(i)
            self.next_inst = (self.next_inst + 1) % self.smdp.num_inst
        assert np.sum([len(entry) for entry in inst_queue]) == len(cur_resp_times)
            
    def get_action_list(self,cur_resp_times,cur_inst_finish_times,inst_queue,allow_bad_states=False,random=False):
        action_list = []
        num_avail_to_serve = len(cur_resp_times)
        avail_inst = np.array(cur_inst_finish_times) <= 0
        inst_to_sched = np.nonzero(avail_inst)[0]
        #sched_order = np.argsort([inst_queue[inst_select][0] for inst_select in inst_to_sched if len(inst_queue[inst_select]) > 0])
        start_idx = 0
        total_serve = 0
        for inst_select in inst_to_sched:
            cur_queue = inst_queue[inst_select]
            try:
                inst_action, num_served = self.inst_pi_obj_list[inst_select].get_action(np.array(cur_resp_times)[cur_queue])
            except Exception as e:
                inst_action = 0
                num_served = 0
                if not allow_bad_states:
                    #print(inst_select,cur_queue)
                    raise e
            cur_served = cur_queue[:num_served]
            start_idx += num_served
            total_serve += len(cur_served)
            action_list.append((inst_action, inst_select, cur_served))
        assert len(inst_to_sched) != 0 or start_idx == 0
        return action_list, total_serve
        
    def commit_action_list(self,cur_resp_times,cur_inst_finish_times,inst_queue,action_list):
        residual_q = 0
        total_serve = 0
        for inst_action, inst_select, cur_served in action_list:
            num_served = len(cur_served)
            self.batch_size_profile.append(num_served)
            residual_q += np.maximum(num_served - len(inst_queue[inst_select]),0)
            inst_queue[inst_select] = inst_queue[inst_select][num_served:]
            self.total_count[inst_select] += num_served
            total_serve += num_served
        
        total_remain = []
        for inst_select in range(len(inst_queue)):
            total_remain += list(inst_queue[inst_select])
        
        sorted_idx = np.sort(total_remain)
        new_idx_map = dict((sorted_idx[i],i) for i in range(len(sorted_idx)))
        for inst_select in range(len(inst_queue)):
            cur_queue = inst_queue[inst_select]
            for i in range(len(cur_queue)):
                cur_queue[i] = new_idx_map[cur_queue[i]]
        
        assert residual_q == 0
        assert np.sum([len(entry) for entry in inst_queue]) == len(cur_resp_times) - total_serve, "fail"
    
    def get_action(self,cur_resp_times,cur_inst_finish_times):
        if len(cur_resp_times) == 0 or all(np.array(cur_inst_finish_times) > 0.):
            return []
        self.sched_call_count += 1
        self.queue_new_queries(cur_resp_times,cur_inst_finish_times,self.inst_queue)
        action_list,total_query = self.get_action_list(cur_resp_times,cur_inst_finish_times,self.inst_queue)
        self.commit_action_list(cur_resp_times,cur_inst_finish_times,self.inst_queue,action_list)
        return action_list

class FullInorderQueuePi:
    def __init__(self,smdp,inst_pi_obj_list,inst_weight,**kwargs):
        self.smdp = smdp
        self.inst_pi_obj_list = inst_pi_obj_list
        self.inst_weight = inst_weight
        self.total_count = np.zeros((smdp.num_inst),dtype=np.int64)
        self.sched_call_count = 0
        self.batch_size_profile = []
    
    def synchronize_state(self,other_pi_obj):
        pass
            
    def get_action_list(self,cur_resp_times,cur_inst_finish_times,allow_bad_states=False,random=False):
        action_list = []
        num_avail_to_serve = len(cur_resp_times)
        avail_inst = np.array(cur_inst_finish_times) <= 0
        inst_to_sched = np.nonzero(avail_inst)[0]
        start_idx = 0
        total_serve = 0
        for inst_select in inst_to_sched:
            try:
                inst_action, num_served = self.inst_pi_obj_list[inst_select].get_action(np.array(cur_resp_times)[total_serve:])
            except Exception as e:
                inst_action = 0
                num_served = 0
                if not allow_bad_states:
                    raise e
            cur_served = [total_serve+i for i in range(num_served)]
            start_idx += num_served
            total_serve += len(cur_served)
            action_list.append((inst_action, inst_select, cur_served))
        assert len(inst_to_sched) != 0 or start_idx == 0
        return action_list, total_serve
        
    def get_action(self,cur_resp_times,cur_inst_finish_times):
        if len(cur_resp_times) == 0 or all(np.array(cur_inst_finish_times) > 0.):
            return []
        self.sched_call_count += 1
        action_list,total_query = self.get_action_list(cur_resp_times,cur_inst_finish_times)
        return action_list
    
class LeastLoadedPi:
    def __init__(self,smdp,inst_pi_obj_list,inst_weight,**kwargs):
        self.smdp = smdp
        self.inst_pi_obj_list = inst_pi_obj_list
        self.inst_weight = inst_weight
        self.total_count = np.zeros((smdp.num_inst),dtype=np.int64)
        self.inst_queue = [[] for i in range(smdp.num_inst)]
        self.running_queue = [[] for i in range(smdp.num_inst)]
        self.sched_call_count = 0
    
    def synchronize_state(self,other_pi_obj):
        other_pi_obj.inst_queue = self.inst_queue
        other_pi_obj.running_queue = self.running_queue
    
    def queue_new_queries(self,cur_resp_times,cur_inst_finish_times,inst_queue):
        if len(cur_resp_times) == 0:
            return
        cur_inst_load = [len(inst_queue[i]) + len(self.running_queue[i]) for i in range(self.smdp.num_inst)]
        cur_queue_size = np.sum([len(entry) for entry in inst_queue])     
        num_new_queue = len(cur_resp_times) - cur_queue_size
        for i in range(cur_queue_size,len(cur_resp_times),1):
            inst_select = np.argmin(cur_inst_load)
            inst_queue[inst_select].append(i)
            cur_inst_load[inst_select] += 1
        assert np.sum([len(entry) for entry in inst_queue]) == len(cur_resp_times)
            
    def get_action_list(self,cur_resp_times,cur_inst_finish_times,inst_queue,allow_bad_states=False,random=False):
        action_list = []
        num_avail_to_serve = len(cur_resp_times)
        avail_inst = np.array(cur_inst_finish_times) <= 0
        inst_to_sched = np.nonzero(avail_inst)[0]
        #sched_order = np.argsort([inst_queue[inst_select][0] for inst_select in inst_to_sched if len(inst_queue[inst_select]) > 0])
        start_idx = 0
        total_serve = 0
        for inst_select in inst_to_sched:
            cur_queue = inst_queue[inst_select]
            try:
                inst_action, num_served = self.inst_pi_obj_list[inst_select].get_action(np.array(cur_resp_times)[cur_queue])
            except Exception as e:
                inst_action = 0
                num_served = 0
                if not allow_bad_states:
                    print(1/self.smdp.rate)
                    print(cur_resp_times)
                    print(len(cur_resp_times),cur_queue)
                    print(start_idx)
                    print(inst_select,cur_queue)
                    raise e
            cur_served = cur_queue[:num_served]
            start_idx += num_served
            total_serve += len(cur_served)
            action_list.append((inst_action, inst_select, cur_served))
        assert len(inst_to_sched) != 0 or start_idx == 0
        return action_list, total_serve
        
    def commit_action_list(self,cur_resp_times,cur_inst_finish_times,inst_queue,action_list):
        residual_q = 0
        total_serve = 0
        for inst_action, inst_select, cur_served in action_list:
            num_served = len(cur_served)
            residual_q += np.maximum(num_served - len(inst_queue[inst_select]),0)
            self.running_queue[inst_select] = inst_queue[inst_select][:num_served]
            inst_queue[inst_select] = inst_queue[inst_select][num_served:]
            self.total_count[inst_select] += num_served
            total_serve += num_served
        
        total_remain = []
        for inst_select in range(len(inst_queue)):
            total_remain += list(inst_queue[inst_select])
        
        sorted_idx = np.sort(total_remain)
        new_idx_map = dict((sorted_idx[i],i) for i in range(len(sorted_idx)))
        for inst_select in range(len(inst_queue)):
            cur_queue = inst_queue[inst_select]
            for i in range(len(cur_queue)):
                cur_queue[i] = new_idx_map[cur_queue[i]]
        
        assert residual_q == 0
        assert np.sum([len(entry) for entry in inst_queue]) == len(cur_resp_times) - total_serve, "fail"
    
    def get_action(self,cur_resp_times,cur_inst_finish_times):
        if len(cur_resp_times) == 0 or all(np.array(cur_inst_finish_times) > 0.):
            return []
        self.sched_call_count += 1
        self.queue_new_queries(cur_resp_times,cur_inst_finish_times,self.inst_queue)
        action_list,total_query = self.get_action_list(cur_resp_times,cur_inst_finish_times,self.inst_queue)
        self.commit_action_list(cur_resp_times,cur_inst_finish_times,self.inst_queue,action_list)
        return action_list

from tqdm import tqdm
import itertools
from multiprocessing import Pool

from simulator import *
from solver import *
from GeneralSMDP import *

class DynamicPi:
    def __init__(self,central_pi_obj_list,rate_list):
        self.central_pi_obj_list = central_pi_obj_list
        self.rate_list = np.array(rate_list)
        self.last_pi_obj = None
    
    def get_action(self,cur_resp_times,cur_inst_finish_times,cur_rate):
        select_mask = self.rate_list >= cur_rate
        if not any(select_mask):
            select_idx = -1
        else:
            select_idx = np.argmax(select_mask)
        cur_pi_obj = self.central_pi_obj_list[select_idx]
        if self.last_pi_obj is not None:
            self.last_pi_obj.synchronize_state(cur_pi_obj)
        action_list = cur_pi_obj.get_action(cur_resp_times,cur_inst_finish_times)
        self.last_pi_obj = cur_pi_obj
        return action_list
    
    def reset_state(self):
        for i in range(len(self.central_pi_obj_list)):
            self.central_pi_obj_list[i].reset_state()

def dynamic_pi_loader(
    inst_sched_constructor,
    central_sched_constructor,
    inst_smdp_constructor,
    smdp,
    exp_dir,
    cur_model_inf_time, 
    cur_model_inf_time_probmass, 
    acc_list, 
    rate_list,
    target_violation_rate,
    std_dev_frac=0, 
    **kwargs
    ):
    cur_rate_list = np.sort(rate_list)
    central_pi_obj_list = []
    success_rate_list = []
    for rate in cur_rate_list:
        try:
            pi_obj = central_pi_loader(
                inst_sched_constructor,
                central_sched_constructor,
                inst_smdp_constructor,
                smdp,
                exp_dir,
                cur_model_inf_time, 
                cur_model_inf_time_probmass, 
                acc_list, 
                rate,
                target_violation_rate,
                std_dev_frac=0, 
                **kwargs
            )
            central_pi_obj_list.append(pi_obj)
            success_rate_list.append(rate)
        except Exception as e:
            pass
            #print('failed!')
            raise e
    assert len(success_rate_list) > 0
    print("policy rate range:",len(success_rate_list))
    return DynamicPi(central_pi_obj_list=central_pi_obj_list,rate_list=success_rate_list)

def central_pi_loader(
    inst_sched_constructor,
    central_sched_constructor,
    inst_smdp_constructor,
    smdp,
    exp_dir,
    cur_model_inf_time, 
    cur_model_inf_time_probmass, 
    acc_list, 
    rate,
    target_violation_rate,
    std_dev_frac=0,
    load_pi=True,
    load_safety_V=False,
    load_offline_calc=False,
    **kwargs):
        
    if smdp.num_inst == 1:
        inst_weight = np.array([1.])
    else:
        inst_weight = np.array([1/smdp.num_inst for entry in range(smdp.num_inst)])

    pi_obj_list = []
    for i in range(smdp.num_inst):
        cur_rate = rate
        #print(local_rate,rate*inst_weight[i],avail_policy_rates)
        #pi_smdp = Batch1SMDP(
        pi_smdp = inst_smdp_constructor(
            rate=cur_rate,
            SLO=smdp.SLO,
            model_inf_time=cur_model_inf_time,
            model_inf_time_probmass=cur_model_inf_time_probmass,
            model_accuracy=acc_list,
            resp_time=None,
            num_inst=smdp.num_inst,
            **kwargs
        )
        if load_pi:
            exp_name = get_exp_fname(exp_dir,cur_rate,target_violation_rate,smdp.SLO,smdp.num_inst)
            #print(exp_name)
            pi = input_pi(exp_name)
            if -1 in pi and len(pi) == 1:
                kwargs["static_action"] = pi[-1]
            if not load_safety_V:
                safety_V_dict = None
            else:
                safety_V_dict = input_V(exp_name+'_safety')
            if not load_offline_calc:
                if i == 0:
                    #safety_dict = {-1:np.sum(get_F_vec(pi,pi_smdp,max_iter=100))}
                    #V_dict = {-1:compute_total_value(pi_smdp,pi,num_iters=100)}
                    safety_dict = {-1:0}
                    V_dict = {-1:1}
            else:
                safety_dict = input_V(exp_name+'_policysafety')
                V_dict = input_V(exp_name)
            pi_obj = inst_sched_constructor(smdp=pi_smdp,pi=pi,safety_V_dict=safety_V_dict,safety_dict=safety_dict,V_dict=V_dict,**kwargs)
        else:
            pi_obj = inst_sched_constructor(smdp=pi_smdp,**kwargs)
        pi_obj_list.append(pi_obj)

    pi_obj = central_sched_constructor(smdp,pi_obj_list,inst_weight=inst_weight,**kwargs)
    return pi_obj

def test_search_policy(env_list,
                       all_rate_list,
                       scheduler_rate_list,
                       exp_dir=None,
                       inst_sched_constructor=None,
                       central_sched_constructor=None,
                       inst_smdp_constructor=None,
                       dynamic_pi_list=None,
                       distribution="poisson",
                       target_violation_rate=None,
                       eval_seconds=5*60,
                       **kwargs):
    total_test_accuracy = []
    total_test_accuracy_list = []
    total_test_violation = []
    total_test_total_query = []
    total_test_pi_obj_list = []
    count = 0
    p_args = []
    for j in tqdm(range(len(env_list))):
        num_inst, cur_model_inf_time, cur_model_inf_time_probmass, acc_list,SLO = env_list[j]
        cur_count = len(all_rate_list)
        test_accuracy = [np.min(acc_list) for i in range(cur_count)]
        test_accuracy_list = [[] for i in range(cur_count)]
        test_violation = [1. for i in range(cur_count)]
        test_total_query = [0 for i in range(cur_count)]
        test_pi_obj_list = [None for i in range(cur_count)]
        for i in range(len(all_rate_list)):
            smdp = GeneralSMDP(
                    rate_list=all_rate_list[i],
                    SLO=SLO,
                    model_inf_time=cur_model_inf_time,
                    model_inf_time_probmass=cur_model_inf_time_probmass,
                    model_accuracy=acc_list,
                    resp_time=None,
                    num_inst=num_inst,
                    max_batch_size=128
            )
            try:
                pi_obj = dynamic_pi_loader(
                    inst_sched_constructor,
                    central_sched_constructor,
                    inst_smdp_constructor,
                    smdp,
                    exp_dir,
                    cur_model_inf_time,
                    cur_model_inf_time_probmass,
                    acc_list,
                    scheduler_rate_list[i],
                    target_violation_rate,
                    **kwargs)
                test_pi_obj_list[i] = pi_obj
                p_args.append((copy.deepcopy(smdp),copy.deepcopy(pi_obj),copy.deepcopy((j,i)),distribution,eval_seconds))
            except Exception as e:
                print('could not load policy for rate',i,all_rate_list[i])
                raise e
        total_test_accuracy.append(test_accuracy)
        total_test_violation.append(test_violation)
        total_test_total_query.append(test_total_query)
        total_test_pi_obj_list.append(test_pi_obj_list)
        total_test_accuracy_list.append(test_accuracy_list)
        count += 1
    
    p=Pool(processes = 4)
    output = p.map(sim_worker,p_args)
    p.close()
    p.join()
    """
    output = []
    for entry in p_args:
        #output.append(sim_worker(entry))
        smdp,pi_obj,idx,distribution = entry
        violation_rate, accuracy, accuracy_list, total_query = get_results(smdp,pi_obj,distribution=distribution)
        print('finished',idx)
        output.append((violation_rate,accuracy,total_query,pi_obj,idx))
    """
    for res in output:
        if res is not None:
            violation_rate, accuracy, accuracy_list, total_query, pi_obj, idx = res
            j,i = idx
            total_test_accuracy[j][i] = accuracy
            total_test_accuracy_list[j][i] = accuracy_list
            total_test_violation[j][i] = violation_rate
            total_test_total_query[j][i] = total_query
            total_test_pi_obj_list[j][i] = pi_obj
        
    return total_test_accuracy, total_test_accuracy_list, total_test_violation, total_test_total_query, total_test_pi_obj_list

def get_save_name(exp_name,target_violation_rate,distribution,trace=None):    
    full_name = exp_name + "_"+ str(target_violation_rate)+"_"+distribution
    full_name_w_trace = full_name +"_trace-"+trace
    print(full_name_w_trace)
    return full_name_w_trace

def save_results(save_name,
                 rate_list,
                 env_list,
                 accuracy_list,
                 violation_list,
                 total_query_list,
                 expected_accuracy_list=None,
                 expected_violation_list=None,
                ):
    if os.path.dirname(save_name) != "":
        os.makedirs(os.path.dirname(save_name),exist_ok=True)
    for i in range(len(env_list)):
        cur_num_inst,_,_,_,cur_SLO = env_list[i]
        for j in range(len(rate_list)):
            tmp_rate = rate_list[j][0]
            cur_str = save_name+'_inst-'+str(cur_num_inst)+'_SLO-'+str(int(1e-3*cur_SLO))+'ms_rate-'+str(tmp_rate)
            f = open(cur_str+'_accuracy.json','w')
            json.dump(accuracy_list[i][j],f,cls=NpEncoder)
            f.close()
            f = open(cur_str+'_violation.json','w')
            json.dump(violation_list[i][j],f,cls=NpEncoder)
            f.close()
            f = open(cur_str+'_query.json','w')
            json.dump(total_query_list[i][j],f,cls=NpEncoder)
            f.close()
            if expected_accuracy_list is not None:
                f = open(cur_str+'_expected_accuracy.json','w')
                json.dump(expected_accuracy_list[i][j],f,cls=NpEncoder)
                f.close()
            if expected_violation_list is not None:
                f = open(cur_str+'_expected_violation.json','w')
                json.dump(expected_violation_list[i][j],f,cls=NpEncoder)
                f.close()
        
def load_results(save_name,rate_list,env_list):
    accuracy_list = []
    violation_list = []
    total_query_list = []
    expected_accuracy_list = None
    expected_violation_list = None
    for i in range(len(env_list)):
        accuracy_list.append([])
        violation_list.append([])
        total_query_list.append([])
        cur_num_inst,_,_,_,cur_SLO = env_list[i]
        for j in range(len(rate_list)):
            tmp_rate = rate_list[j][0]
            cur_str = save_name+'_inst-'+str(cur_num_inst)+'_SLO-'+str(int(1e-3*cur_SLO))+'ms_rate-'+str(tmp_rate)
            f = open(cur_str+'_accuracy.json','r')
            val = json.load(f)
            accuracy_list[i].append(val)
            f.close()
            f = open(cur_str+'_violation.json','r')
            val = json.load(f)
            violation_list[i].append(val)
            f.close()
            f = open(cur_str+'_query.json','r')
            val = json.load(f)
            total_query_list[i].append(val)
            f.close()
    return accuracy_list, violation_list, total_query_list, rate_list, expected_accuracy_list, expected_violation_list

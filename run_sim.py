import numpy as np
import argparse
import scipy.stats
from tqdm import tqdm
import itertools
from multiprocessing import Pool

from solver import *
from distribution import *
from VarBatchSMDP import *
from profile_utils import *
from OfflineCentralPolicy import *
from model_selector import *
from twitter_trace import trace_loader
from simulator_script import *

def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulates model selection policies under stochastic query arrivals",
    )

    parser.add_argument(
        "--m",
        required=True,
        help="chose from one of the model selection methods: RAMSIS, JF, or MS",
    )
    
    parser.add_argument(
        "--trace",
        required=True,
        help="chose the production trace or constant load trace: real or constant",
    )
    
    parser.add_argument(
        "--task",
        required=False,
        default="image",
        help="chose the inference task: image or text",
    )
    
    parser.add_argument(
        "--load",
        required=False,
        default="",
        help="list of query load in queries per second that specify the pre-computed model selection policies used in the simulation by the method",
    )
        
    parser.add_argument(
        "--SLO",
        required=False,
        default="150",
        help="list of latency SLOs in milliseconds",
    )

    parser.add_argument(
        "--worker",
        required=False,
        default="",
        help="list of number of workers",
    )
        
    args = parser.parse_args()
    method = args.m
    trace_mode = args.trace
    task = args.task
    if trace_mode == "real":
        if args.load == "":
            rates = [200*(i+1)/1000000 for i in range(20)]
        else:    
            rates = [float(entry)/1000000 for entry in args.load.split(",")]
        if args.worker == "":
            num_inst_list = [10*(i+1) for i in range(5,8)]
        else:    
            num_inst_list = [int(entry) for entry in args.worker.split(",")]
    elif trace_mode == "constant":
        if args.load == "":
            rates = [400*(i+1)/1000000 for i in range(10)]
        else:
            rates = [float(entry)/1000000 for entry in args.load.split(",")]
        if args.worker == "":
            num_inst_list = [60]
        else:
            num_inst_list = [int(entry) for entry in args.worker.split(",")]
    else:
        assert False, "invalid trace option!"
    SLO_list = [float(entry)*1000 for entry in args.SLO.split(",")]
    return method, trace_mode, task, rates, SLO_list, num_inst_list

method, trace_mode, task, rates, SLO_list, num_inst_list = parse_args()

if task == "image":
    cur_models = ['shufflenet_v2_x0_5',
        'mobilenet_v3_large',
        'efficientnet_b0',
        'efficientnet_b2',
        'efficientnet_b3',
        'efficientnet_b4',
        'efficientnet_v2_s',
        'efficientnet_v2_m',
        'efficientnet_v2_l',
    ]
    cur_home_dir = "image_cloud_cpu"
elif task == "text":
    cur_models = [
        "bert_tiny",
        "bert_mini",
        "bert_small",
        "bert_medium",
        "bert_base",
    ]
    cur_home_dir = "text_cloud_cpu"
else:
    assert False, "ERROR: specified task not found!"

cur_distribution="poisson"
cur_save_dir = "results/"+cur_home_dir
cur_policy_dir = "policy_gen/"+cur_home_dir

profile_batch_size = 32
max_batch_size=32

all_rate_list = []
cur_rates = rates
if trace_mode == "constant":
    eval_seconds = 30
    for entry in cur_rates:
        all_rate_list.append([entry])
elif trace_mode == "real":
    eval_seconds = 60*5
    all_rate_list = [trace_loader.get_real_trace("twitter_trace/twitter_04_25_norm.txt")]
    
if trace_mode == "constant":
    scheduler_rate_list = all_rate_list.copy()
else:
    scheduler_rate_list = [cur_rates.copy() for i in range(len(all_rate_list))]    

cur_model_inf_time, cur_model_inf_time_probmass, acc_list = get_pt_latency_dist_accuracy_list(
            home_dir=cur_home_dir,
            models=cur_models,
            max_batch_size=profile_batch_size,
            percentile=95,
)

env_list = []
for num_inst in num_inst_list:
    for SLO in SLO_list:
        env_list.append((num_inst,cur_model_inf_time,cur_model_inf_time_probmass,acc_list,SLO))

def calc_offline_accuracy(pi_obj):
    arr = []
    rate_weight = np.array(pi_obj.rate_list)/np.sum(pi_obj.rate_list)
    for central_pi_obj in pi_obj.central_pi_obj_list:
        cur_arr = []
        for inst_pi_obj in central_pi_obj.inst_pi_obj_list:
            cur_arr.append(inst_pi_obj.V_dict[-1])
        cur_res = np.sum(np.array(cur_arr)*np.array(central_pi_obj.inst_weight))
        arr.append(cur_res)
    res = np.sum(np.array(arr)*rate_weight)
    return res

def calc_offline_violation(pi_obj):
    arr = []
    rate_weight = np.array(pi_obj.rate_list)/np.sum(pi_obj.rate_list)
    for central_pi_obj in pi_obj.central_pi_obj_list:
        cur_arr = []
        for inst_pi_obj in central_pi_obj.inst_pi_obj_list:
            cur_arr.append(inst_pi_obj.safety_dict[-1])
        cur_res = np.sum(np.array(cur_arr)*np.array(central_pi_obj.inst_weight))
        arr.append(cur_res)
    res = np.sum(np.array(arr)*rate_weight)
    return res   

def get_offline_metrics(pi_obj_list):
    expected_violation_list = []
    expected_accuracy_list = []
    for cur_idx in range(len(pi_obj_list)):
        expected_violation_list.append([calc_offline_violation(entry) for entry in pi_obj_list[cur_idx] if entry is not None])
        expected_accuracy_list.append([calc_offline_accuracy(entry) for entry in pi_obj_list[cur_idx] if entry is not None])
    return expected_accuracy_list,expected_violation_list

if __name__ ==  '__main__':
    if method == "MS":
        target_violation_rate = 1e-2
        cur_name = "_ModelSwitching"
        save_name = cur_name    
        total_ModelSwitching_accuracy, total_ModelSwitching_accuracy_list, total_ModelSwitching_violation, total_ModelSwitching_query, total_ModelSwitching_pi_obj_list = test_search_policy(
            env_list=env_list,
            all_rate_list=all_rate_list,
            scheduler_rate_list=scheduler_rate_list,
            exp_dir=cur_policy_dir+cur_name,
            distribution=cur_distribution,
            inst_sched_constructor=StaticInstPi,
            #central_sched_constructor=CentralBalancePi,
            #central_sched_constructor=DeterministicRoundRobinPi,
            #central_sched_constructor=LeastLoadedPi,
            central_sched_constructor=FullInorderQueuePi,
            inst_smdp_constructor=VarBatchSMDP,
            target_violation_rate=target_violation_rate,
            batch_delay=False,
            load_pi=True,
            max_batch_size=profile_batch_size,  
            eval_seconds=eval_seconds,
        )

        save_results(save_name=get_save_name(cur_save_dir+save_name,target_violation_rate,cur_distribution,trace_mode),
                     rate_list=all_rate_list,
                     env_list=env_list,
                     accuracy_list=total_ModelSwitching_accuracy,
                     violation_list=total_ModelSwitching_violation,
                     total_query_list=total_ModelSwitching_query,
                    )
    elif method == "JF":
        cur_name = "_jellyfish"
        total_jellyfish_accuracy, total_jellyfish_accuracy_list, total_jellyfish_violation, total_jellyfish_query, total_jellyfish_pi_obj_list = test_search_policy(
            env_list=env_list,
            all_rate_list=all_rate_list,
            scheduler_rate_list=scheduler_rate_list,        
            distribution=cur_distribution,
            inst_sched_constructor=JellyfishInstPi,
            #central_sched_constructor=LeastLoadedPi,
            #central_sched_constructor=DeterministicRoundRobinPi,
            central_sched_constructor=FullInorderQueuePi,
            inst_smdp_constructor=VarBatchSMDP,
            load_offline_calc=True,
            load_pi=False,
            max_batch_size=profile_batch_size,
            batch_delay=False,
            eval_seconds=eval_seconds,
        )
        save_results(save_name=get_save_name(cur_save_dir+cur_name,None,cur_distribution,trace_mode),
                     rate_list=all_rate_list,
                     env_list=env_list,
                     accuracy_list=total_jellyfish_accuracy,
                     violation_list=total_jellyfish_violation,
                     total_query_list=total_jellyfish_query,
        )
    elif method == "RAMSIS":
        cur_name = '_RAMSIS'
        cur_res_name = cur_name
        fixed_interval_time=True
        num_interval = 100        
        total_RAMSIS_accuracy, total_RAMSIS_accuracy_list, total_RAMSIS_violation, total_RAMSIS_query, total_RAMSIS_pi_obj_list = test_search_policy(
            env_list=env_list,
            all_rate_list=all_rate_list,
            scheduler_rate_list=scheduler_rate_list,
            exp_dir=cur_policy_dir+cur_name,
            distribution=cur_distribution,
            inst_sched_constructor=PlainInstPi,
            #central_sched_constructor=RoundRobinBalancePi,
            #central_sched_constructor=LeastLoadedPi,
            central_sched_constructor=DeterministicRoundRobinPi,
            inst_smdp_constructor=VarBatchSMDP,
            target_violation_rate=None,
            load_offline_calc=True,
            max_batch_size=profile_batch_size,
            fixed_interval_time=fixed_interval_time,
            num_interval=num_interval,
            eval_seconds=eval_seconds,
        )
        RAMSIS_expected_accuracy,RAMSIS_expected_violation = get_offline_metrics(total_RAMSIS_pi_obj_list)
        save_results(save_name=get_save_name(cur_save_dir+cur_res_name,None,cur_distribution,trace_mode),
                     rate_list=all_rate_list,
                     env_list=env_list,
                     accuracy_list=total_RAMSIS_accuracy,
                     violation_list=total_RAMSIS_violation,
                     total_query_list=total_RAMSIS_query,
                     expected_accuracy_list=RAMSIS_expected_accuracy,
                     expected_violation_list=RAMSIS_expected_violation,
                    )
    else:
        assert False, "ERROR: invalid model selection method!"
        
print("script complete!")
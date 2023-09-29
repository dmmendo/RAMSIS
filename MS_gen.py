import numpy as np
import argparse
import scipy.stats
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import itertools
from multiprocessing import Pool

from solver import *
from distribution import *
from profile_utils import *
from VarBatchSMDP import *
from model_selector import *
from OfflineCentralPolicy import *
from simulator_script import *

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generates RAMSIS model selection policies",
    )

    parser.add_argument(
        "--task",
        required=False,
        default="image",
        help="chose the inference task",
    )
    
    parser.add_argument(
        "--load",
        required=False,
        default=",".join([str(200*(i+1)) for i in range(20)]),
        help="list of query load in queries per second",
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
        default=",".join([str(10*(i+1)) for i in range(5,8)]),
        help="list of number of workers",
    )
        
    args = parser.parse_args()
    task = args.task
    rates = [float(entry)/1000000 for entry in args.load.split(",")]
    SLO_list = [float(entry)*1000 for entry in args.SLO.split(",")]
    num_inst_list = [int(entry) for entry in args.worker.split(",")]
    return task, rates, SLO_list, num_inst_list

task, rates, SLO_list, num_inst_list = parse_args()
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

cur_save_dir = "policy_gen/"+cur_home_dir
profile_batch_size = 32
max_batch_size=32
cur_model_inf_time, cur_model_inf_time_probmass, acc_list = get_pt_latency_dist_accuracy_list(
        home_dir=cur_home_dir,
        models=cur_models,
        max_batch_size=profile_batch_size,
        percentile=95)

trace_mode = "constant"

target_violation_rate = 1e-2
cur_distribution="poisson"

all_rate_list = []
cur_rates = rates
#cur_rates = get_avail_policy_rates()
if trace_mode == "constant":
    for entry in cur_rates:
        all_rate_list.append([entry])
elif trace_mode == "real":
    all_rate_list = [get_real_trace("twitter_trace/twitter_04_25_norm.txt")]
    
if trace_mode == "constant":
    scheduler_rate_list = all_rate_list.copy()
else:
    scheduler_rate_list = [cur_rates.copy() for i in range(len(all_rate_list))]
    
env_list = []
for num_inst in num_inst_list:
    for SLO in SLO_list:
        env_list.append((num_inst,cur_model_inf_time,cur_model_inf_time_probmass,acc_list,SLO))
        
batch_delay=False
eval_seconds = 5*60
cur_name = "ModelSwitching"
if __name__ ==  '__main__':
    total_ModelSwitching_accuracy = []
    total_ModelSwitching_accuracy_list = []
    total_ModelSwitching_violation = []
    total_ModelSwitching_query = []
    total_ModelSwitching_pi_obj_list = []
    for i in range(1,len(acc_list)+1):
        cur_accuracy,cur_accuracy_list, cur_violation, cur_query, cur_pi_obj_list = test_search_policy(
            env_list=env_list,
            all_rate_list=all_rate_list,
            scheduler_rate_list=scheduler_rate_list,
            distribution=cur_distribution,
            inst_sched_constructor=StaticInstPi,
            central_sched_constructor=FullInorderQueuePi,
            inst_smdp_constructor=VarBatchSMDP,
            load_offline_calc=True,
            load_pi=False,
            static_action=i,
            batch_delay=batch_delay,
            eval_seconds=eval_seconds,
        )
        total_ModelSwitching_accuracy.append(cur_accuracy)
        total_ModelSwitching_accuracy_list.append(cur_accuracy_list)
        total_ModelSwitching_violation.append(cur_violation)
        total_ModelSwitching_query.append(cur_query)
        total_ModelSwitching_pi_obj_list.append(cur_pi_obj_list)

    success_mask = np.array(total_ModelSwitching_violation) < target_violation_rate
    score = success_mask * np.array(total_ModelSwitching_accuracy)
    for i in range(len(env_list)):
        num_inst, cur_model_inf_time, cur_model_inf_time_probmass, acc_list,SLO = env_list[i]
        for j in range(len(rates)):
            action = np.argmax(score[:,i,j]) + 1
            pi = {-1:action}
            fname = get_exp_fname(cur_save_dir+"_"+cur_name,rates[j],target_violation_rate,SLO,num_inst)
            print(fname)
            os.makedirs(os.path.dirname(fname),exist_ok=True)
            output_pi(fname,pi)

print("script complete!")

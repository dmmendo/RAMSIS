import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
import argparse
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import copy

from solver import *
from distribution import *
from VarBatchSMDP import *
from profile_utils import *

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

arrival_dist = "erlang"

target_violation_rate = None

env_list = []
for num_inst in num_inst_list:
    for SLO in SLO_list:
        env_list.append((num_inst,cur_model_inf_time,cur_model_inf_time_probmass,acc_list,SLO))
        
max_action_batch=None #default: None
fixed_interval_time=True #default: True
num_interval = 100 #default: 100

if __name__ ==  '__main__': 
    p_args = []
    for i in range(len(env_list)):
        num_inst,cur_model_inf_time,cur_model_inf_time_probmass,acc_list,SLO = env_list[i]
        for i in range(len(rates)):
            smdp = VarBatchSMDP(
                rate=rates[i],
                SLO=SLO,
                model_inf_time=cur_model_inf_time,
                model_inf_time_probmass=cur_model_inf_time_probmass,
                model_accuracy=acc_list,
                resp_time=None,
                arrival_dist=arrival_dist,
                max_batch_size=max_batch_size,
                num_inst=num_inst,
                max_action_batch=max_action_batch,
                #prob_thresh=1e-9,
                prob_thresh=0.,
                fixed_interval_time=fixed_interval_time,
                num_interval = num_interval,
            )
            exp_name = get_exp_fname(
                exp_dir=cur_save_dir+"_RAMSIS",
                qrate=rates[i],
                target_violation=target_violation_rate,
                SLO=SLO,
                num_inst=num_inst,
            )
            p_args.append((exp_name,copy.deepcopy(smdp),100,target_violation_rate))
    print(len(smdp.resp_time))
   
    p=Pool(processes = 4)
    output = p.map(valueIteration_worker,p_args)
    p.close()
    p.join()

print("script complete!")

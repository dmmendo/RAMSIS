import numpy as np
import argparse
import scipy.stats
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import itertools
from multiprocessing import Pool
from tabulate import tabulate
from profile_utils import *
from twitter_trace import trace_loader
from simulator_script import *

def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulates model selection policies under stochastic query arrivals",
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
        help="latency SLO in milliseconds",
    )

    parser.add_argument(
        "--worker",
        required=False,
        default="",
        help="list of number of workers",
    )
        
    args = parser.parse_args()
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
    SLO_list = [float(args.SLO)*1000]
    return trace_mode, task, rates, SLO_list, num_inst_list

trace_mode, task, rates, SLO_list, num_inst_list = parse_args()

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
    
profile_batch_size = 32
max_batch_size=32
cur_save_dir = "results/"+cur_home_dir
cur_distribution="poisson"
all_rate_list = []
cur_rates = rates
if trace_mode == "constant":
    for entry in cur_rates:
        all_rate_list.append([entry])
elif trace_mode == "real":
    all_rate_list = [trace_loader.get_real_trace("twitter_trace/twitter_04_25_norm.txt")]
    
cur_model_inf_time, cur_model_inf_time_probmass, acc_list = get_pt_latency_dist_accuracy_list(
    home_dir=cur_home_dir,
    models=cur_models,
    max_batch_size=profile_batch_size)

env_list = []
for num_inst in num_inst_list:
    for SLO in SLO_list:
        env_list.append((num_inst,cur_model_inf_time,cur_model_inf_time_probmass,acc_list,SLO))
        
cur_jellyfish_policy_dir = cur_save_dir + "_jellyfish"
cur_RAMSIS_policy_dir = cur_save_dir+"_RAMSIS"
cur_ModelSwitching_policy_dir = cur_save_dir+"_ModelSwitching"

target_violation_rate = 1e-2
total_ModelSwitching_accuracy, total_ModelSwitching_violation, total_ModelSwitching_query, ModelSwitching_ratelist,_,_ = load_results(
    save_name=get_save_name(cur_ModelSwitching_policy_dir,target_violation_rate,cur_distribution,trace_mode),
    rate_list=all_rate_list,
    env_list=env_list,
)

total_jellyfish_accuracy, total_jellyfish_violation,_,_,_,_ = load_results(
    save_name=get_save_name(cur_jellyfish_policy_dir,None,cur_distribution,trace_mode),
    rate_list=all_rate_list,
    env_list=env_list,
)

total_RAMSIS_accuracy, total_RAMSIS_violation, total_RAMSIS_query, RAMSIS_rate_list, RAMSIS_expected_accuracy,RAMSIS_expected_violation = load_results(
    save_name=get_save_name(cur_RAMSIS_policy_dir,None,cur_distribution,trace_mode),
    rate_list=all_rate_list,
    env_list=env_list,
)

def plot_accuracy_vs_qrate(name,accuracy_list,violation_list,violation_target=None,manual_plot_size=None,**kwargs):
    if violation_target is not None:
        mask = np.array(violation_list) > violation_target
        if any(mask):
            plot_size = np.argmax(mask)
        else:
            plot_size = len(accuracy_list)
    else:
        plot_size = len(accuracy_list)
    if manual_plot_size is not None:
        plot_size = manual_plot_size
    x = 1e6*np.array(rates[:plot_size])
    y = np.array(accuracy_list[:plot_size])*100
    plt.plot(x,y,label=name,**kwargs)
    return x, y
    
def plot_violation_vs_qrate(name,accuracy_list,violation_list,target_violation_rate=None,manual_plot_size=None,**kwargs):
    if target_violation_rate is not None:
        mask = np.array(violation_list) > target_violation_rate
        if any(mask):
            plot_size = np.argmax(mask)
        else:
            plot_size = len(violation_list)
    else:
        plot_size = len(violation_list)
    if manual_plot_size is not None:
        plot_size = manual_plot_size
    y_data = 100*np.array(violation_list[:plot_size])
    x = 1e6*np.array(rates[:plot_size])
    plt.plot(x,y_data,label=name,**kwargs)
    return x, y_data

    
def plot_accuracy_vs_inst(name,accuracy_list,violation_list,cur_SLO,rate_idx,accuracy_target=None,violation_target=None,env_list=env_list,**kwargs):
    data_to_plot = []
    x_list = []
    prev_valid = False
    for i in range(len(env_list)):
        num_inst,cur_model_inf_time,cur_model_inf_time_probmass,acc_list,SLO = env_list[i]
        if SLO == cur_SLO:
            if (violation_target is None or violation_list[i][rate_idx] < violation_target) \
                and (accuracy_target is None or accuracy_list[i][rate_idx] >= accuracy_target):
                data_to_plot.append(accuracy_list[i][rate_idx])
                x_list.append(num_inst)
                #prev_valid = True
            elif prev_valid:
                data_to_plot.append(data_to_plot[-1])
                x_list.append(num_inst)
            else:
                prev_valid = False
    data_to_plot = (np.array(data_to_plot)*100)
    plt.plot(x_list,data_to_plot,label=name,**kwargs)
    return np.array(x_list), np.array(data_to_plot)
    
def plot_violation_vs_inst(name,accuracy_list,violation_list,cur_SLO,rate_idx,env_list=env_list,**kwargs):
    data_to_plot = []
    x_list = []
    for i in range(len(env_list)):
        num_inst,cur_model_inf_time,cur_model_inf_time_probmass,acc_list,SLO = env_list[i]
        if SLO == cur_SLO:
            data_to_plot.append(violation_list[i][rate_idx])
            x_list.append(num_inst)
    plt.plot(x_list,100*np.array(data_to_plot),label=name,**kwargs)
    return np.array(x_list), 100*np.array(data_to_plot)
    
if trace_mode == "real":    
    cur_SLO = SLO_list[0] #0,1,2
    cur_rate_idx =0 #1,3,6
    violation_target = None
    accuracy_target = None

    inst_arr,jellyfish_acc = plot_accuracy_vs_inst('Jellyfish+',total_jellyfish_accuracy, total_jellyfish_violation,cur_SLO,rate_idx=cur_rate_idx,accuracy_target=accuracy_target,violation_target=violation_target,alpha=0.5,color='red',linewidth=4,marker='s')
    inst_arr,MS_acc = plot_accuracy_vs_inst('ModelSwitching',total_ModelSwitching_accuracy, total_ModelSwitching_violation,cur_SLO,rate_idx=cur_rate_idx,accuracy_target=accuracy_target,violation_target=violation_target,alpha=0.5,color='green',marker='v',linewidth=4,linestyle='dashdot')
    inst_arr,RAMSIS_acc = plot_accuracy_vs_inst('RAMSIS',total_RAMSIS_accuracy, total_RAMSIS_violation,cur_SLO,rate_idx=cur_rate_idx,accuracy_target=accuracy_target,violation_target=violation_target,alpha=0.5,marker='o',linewidth=2,markersize=8,color='blue')

    plt.xlabel('# workers')
    plt.ylabel('accuracy %')
    plt.title('latency SLO='+str(int(cur_SLO/1000))+'ms')#+', load='+str(1e6*rates[cur_rate_idx])+"qps")
    print("SLO violation rate="+str(violation_target))
    plt.xticks(np.arange(60,80+1,20))
    plt.legend()
    plt.savefig(task+'_'+trace_mode+'_accuracy')
    plt.clf()      
    cur_SLO = SLO_list[0]
    cur_rate_idx = 0

    inst_arr,jellyfish_violation = plot_violation_vs_inst('Jellyfish+',total_jellyfish_accuracy, total_jellyfish_violation,cur_SLO,rate_idx=cur_rate_idx,alpha=0.5,color='red',linewidth=4,marker='s')
    inst_arr,MS_violation = plot_violation_vs_inst('ModelSwitching',total_ModelSwitching_accuracy, total_ModelSwitching_violation,cur_SLO,rate_idx=cur_rate_idx,alpha=0.5,color='green',marker='v',linewidth=4,linestyle='dashdot')
    inst_arr,RAMSIS_violation = plot_violation_vs_inst('RAMSIS',total_RAMSIS_accuracy, total_RAMSIS_violation,cur_SLO,rate_idx=cur_rate_idx,alpha=0.5,marker='o',linewidth=2,markersize=8,color='blue')

    plt.xlabel('# workers')
    plt.ylabel('violation %')
    plt.title('latency SLO='+str(int(cur_SLO/1000))+'ms')#+' qrate='+str(1e6*rates[cur_rate_idx]))
    plt.legend()

    plt.savefig(task+'_'+trace_mode+'_violation')
elif trace_mode == "constant":
    cur_SLO = SLO_list[0] # 0, 1, 2
    cur_num_inst = num_inst_list[0] #text: 20, image: 60
    manual_plot_size = None
    violation_target = 100*1e-2
    for i in range(len(env_list)):
        num_inst,cur_model_inf_time,cur_model_inf_time_probmass,acc_list,SLO = env_list[i]
        if num_inst == cur_num_inst and SLO == cur_SLO:
            num_inst,cur_model_inf_time,cur_model_inf_time_probmass,acc_list,SLO = env_list[i]

            qrate_arr,jellyfish_acc = plot_accuracy_vs_qrate('Jellyfish+',total_jellyfish_accuracy[i], total_jellyfish_violation[i],manual_plot_size=manual_plot_size,violation_target=violation_target,alpha=0.5,color='red',linewidth=4,marker='s')
            qrate_arr,MS_acc = plot_accuracy_vs_qrate('ModelSwitching',total_ModelSwitching_accuracy[i], total_ModelSwitching_violation[i],manual_plot_size=manual_plot_size,violation_target=violation_target,alpha=0.5,color='green',marker='v',linewidth=4,linestyle='dashdot')
            qrate_arr,RAMSIS_acc = plot_accuracy_vs_qrate('RAMSIS',total_RAMSIS_accuracy[i], total_RAMSIS_violation[i],manual_plot_size=manual_plot_size,violation_target=violation_target,alpha=0.5,marker='o',linewidth=2,markersize=8,color='blue')

    plt.xlabel('load')
    plt.ylabel('accuracy %')
    #plt.title('# workers='+str(cur_num_inst)+', SLO='+str(int(cur_SLO/1000))+'ms')
    #plt.title('# workers='+str(cur_num_inst))
    plt.title('SLO='+str(int(cur_SLO/1000))+'ms')
    plt.legend()

    plt.savefig(task+'_'+trace_mode+'_accuracy')
    plt.clf()
    cur_SLO = SLO_list[0]
    cur_num_inst = num_inst_list[0]
    manual_plot_size = None
    for i in range(len(env_list)):
        num_inst,cur_model_inf_time,cur_model_inf_time_probmass,acc_list,SLO = env_list[i]
        if num_inst == cur_num_inst and SLO == cur_SLO:
            num_inst,cur_model_inf_time,cur_model_inf_time_probmass,acc_list,SLO = env_list[i]
            qrate_arr,jellyfish_violation = plot_violation_vs_qrate('Jellyfish+',total_jellyfish_accuracy[i], total_jellyfish_violation[i],manual_plot_size=manual_plot_size,alpha=0.5,color='red',linewidth=4,marker='s')
            qrate_arr,MS_violation = plot_violation_vs_qrate('ModelSwitching',total_ModelSwitching_accuracy[i], total_ModelSwitching_violation[i],manual_plot_size=manual_plot_size,alpha=0.5,color='green',marker='v',linewidth=4,linestyle='dashdot')
            qrate_arr,RAMSIS_violation = plot_violation_vs_qrate('RAMSIS',total_RAMSIS_accuracy[i], total_RAMSIS_violation[i],manual_plot_size=manual_plot_size,alpha=0.5,marker='o',linewidth=2,markersize=8,color='blue')        
    plt.xlabel('load')
    plt.ylabel('violation %')
    #plt.title('num_inst='+str(cur_num_inst)+' latency SLO='+str(int(cur_SLO/1000))+'ms')
    plt.legend()

    plt.savefig(task+'_'+trace_mode+'_violation')
    
RAMSIS_jellyfish_acc = RAMSIS_acc - jellyfish_acc
RAMSIS_ModelSwitching_acc = RAMSIS_acc - MS_acc

if trace_mode == "real":
    x = inst_arr
    header_list = ['# workers','Jellyfish','ModelSwitching','RAMSIS']
elif trace_mode == "constant":
    x = qrate_arr
    header_list = ['load (QPS)','Jellyfish','ModelSwitching','RAMSIS']

print('Accuracy % for task:',task,' latency SLO:',int(cur_SLO/1000),'ms trace:',trace_mode,' (higher is better)') 
print(tabulate(list(zip(x,jellyfish_acc,MS_acc,RAMSIS_acc)),headers=header_list))
print()
print('Latency SLO violation % for task:',task,' latency SLO:',int(cur_SLO/1000),'ms trace:',trace_mode)
print(tabulate(list(zip(x,jellyfish_violation,MS_violation,RAMSIS_violation)),headers=header_list))
print()
print('average accuracy % increase for RAMSIS vs. Jellyfish:',"{:.2f}".format(np.mean(RAMSIS_jellyfish_acc)))
print('highest accuracy % increase for RAMSIS vs. Jellyfish:',"{:.2f}".format(np.max(RAMSIS_jellyfish_acc)))
print('average accuracy % increase for RAMSIS vs. ModelSwitching:',"{:.2f}".format(np.mean(RAMSIS_ModelSwitching_acc)))
print('highest accuracy % increase for RAMSIS vs. ModelSwitching:',"{:.2f}".format(np.max(RAMSIS_ModelSwitching_acc)))

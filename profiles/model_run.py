import torch
import time
from tqdm import tqdm
import numpy as np
import json
import multiprocessing as mp
from image_models_loader import *
import os

def profile_model(model,preprocess,data_input_fnc,max_batch_size):
    runtimes = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for bsize in range(1,max_batch_size+1):
        cur_list = []
        print('profiler running batch size',bsize)
        for i in tqdm(range(5)):
            data = data_input_fnc(bsize)
            batch = preprocess(data).to(device)
            prediction = model(batch)
        for t in tqdm(range(100)):
            start_t = time.time()
            data = data_input_fnc(bsize)
            batch = preprocess(data).to(device)
            prediction = model(batch)
            end_t = time.time()
            cur_list.append(end_t - start_t)
        runtimes.append(cur_list)
    print('profiler finished!')
    return runtimes

def run_model_until_done_worker(queue,num_models,model_load_fnc,data_input_fnc):
    model, preprocess = model_load_fnc()
    batch = preprocess(data_input_fnc(1))
    queue.put(0) #signal model loaded
    while queue.qsize() < num_models:
        pass
    print('helper running!')
    queue.put(0) #signal running model
    while queue.qsize() < 2*num_models:
        prediction = model(batch)
    print('helper_finished!')
    
def model_profile_worker(queue, num_models, max_batch_size, model_load_fnc, data_input_fnc):
    while queue.qsize() < num_models-1: #wait for all co-located models to load
        pass
    print('about to load model!')
    model, preprocess = model_load_fnc()
    print('running test!')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch = preprocess(data_input_fnc(1)).to(device)
    try:
        for i in tqdm(range(10)):
            prediction = model(batch)
    except Exception as e:
        print(e)
    print('profiler finished loading model!')
    queue.put(0) #signal profile model is loaded
    while queue.qsize() < 2*num_models-1: #wait for all co-located models to start running
        pass
    runtimes = profile_model(model,preprocess,data_input_fnc,max_batch_size)
    queue.put(runtimes)

def save_results(save_name,results):
    dir_name = os.path.dirname(save_name)
    if dir_name != '':
        os.makedirs(dir_name,exist_ok=True)
    f = open(save_name+'.json','w')
    json.dump(results,f)
    f.close()

def results_loader(save_name):
    f = open(save_name+'.json','r')
    results = json.load(f)
    f.close()
    return results
    
def get_save_str(dir_name,model_name,num_models,batch_size):
    prefix = os.path.join(os.path.join(dir_name,model_name),model_name)
    res = prefix+"_models-"+str(num_models)+"_batchsize-"+str(batch_size)
    if not os.path.exists(res+'.json'):
        res = prefix+"_bsize-"+str(batch_size)+"_profile"
    return res
        
def run_experiment(dir_name,model_name,model_loader,data_loader,num_models,max_batch_size):
    # create a shared event object
    queue = mp.Queue()
    # create a suite of processes
    processes = []
    for i in range(num_models):
        if i == 0:
            processes.append(mp.Process(target=model_profile_worker, args=(queue, num_models, max_batch_size, model_loader, data_loader)))
        else:
            processes.append(mp.Process(target=run_model_until_done_worker, args=(queue, num_models, load_resnet50, get_image_data_input)))
    # start all processes
    for process in processes:
        process.start()

    for process in processes:
        process.join()
    while not queue.empty():
        profile_results = queue.get()
    assert max_batch_size == len(profile_results)
    for i in range(max_batch_size):
        exp_name = get_save_str(dir_name,model_name,num_models,i+1)
        save_results(exp_name,profile_results[i]) 
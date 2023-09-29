import numpy as np
import json
import ast
import os
import time
from distribution import *

def my_valueIteration_vectorized(smdp,max_iter=float('inf')):
    idx_to_state_map = {}
    state_to_idx_map = {}
    
    count = 0
    for state in smdp.states():
        idx_to_state_map[count] = state
        state_to_idx_map[state] = count
        count += 1
    
    V = np.zeros((len(state_to_idx_map)))
    V_count = np.zeros((len(state_to_idx_map)))
    V_avg = np.zeros((len(state_to_idx_map)))
    
    def Q_vectorized(state,action,V,V_count):
        successor_list, prob_list, reward_list = smdp.succProbReward_vectorized(state,action)
        
        num_served = smdp.get_num_served(state,action)
        queue_size = smdp.get_queue_size(state)
        val = prob_list.dot(reward_list*queue_size + smdp.discount()*V)
        val_count = prob_list.dot(num_served + V_count)
        if val_count > 0:
            return val/val_count, val, val_count
        else:
            return 0., 0., 0.
    
    #relative_idx = np.random.randint(len(V))
    num_iter = 0
    while True:
        num_iter += 1
        newV = V.copy()
        newV_count = V_count.copy()
        newV_avg = V_avg.copy()
        iter_diff = 0
        for state in smdp.states():
            if smdp.isEnd(state):
                newV[state_to_idx_map[state]] = 0
            else:
                actions = smdp.actions(state)
                scores = [Q_vectorized(state,action,V,V_count) for action in actions]
                this_action = np.argmax([entry[0] for entry in scores])
                newV[state_to_idx_map[state]] = scores[this_action][1]
                newV_count[state_to_idx_map[state]] = scores[this_action][2]
                newV_avg[state_to_idx_map[state]] = scores[this_action][0]
        #min_diff = np.min(newV)
        #newV -= min_diff
        iter_diff = np.max(np.abs(V_avg-newV_avg))
        V = newV
        V_count = newV_count
        V_avg = newV_avg
        if num_iter >= max_iter or iter_diff < 1e-6:
            break
        #else:
        #    print(iter_diff)
        #if num_iter == 1:
        #    print('finished first iter!')
    #if num_iter >= max_iter:
    #    print('reached max num iterations!')
    #else:
    #    print('converged!')
    
    pi = {}
    resV = {}
    for state in smdp.states():
        resV[state] = V[state_to_idx_map[state]]
        if smdp.isEnd(state):
            pi[state] = 'none'
        else:
            actions = smdp.actions(state)
            pi[state] = actions[np.argmax([Q_vectorized(state,action,V,V_count)[0] for action in actions])]
    return resV, pi

def valueIteration_vectorized(smdp,max_iter=float('inf')):
    idx_to_state_map = {}
    state_to_idx_map = {}
    
    count = 0
    for state in smdp.states():
        idx_to_state_map[count] = state
        state_to_idx_map[state] = count
        count += 1
    
    V = np.zeros((len(state_to_idx_map)))
    
    def Q_vectorized(state,action,V):
        successor_list, prob_list, reward_list = smdp.succProbReward_vectorized(state,action)
        val = prob_list.dot(reward_list + smdp.discount()*V)
        return val
    
    relative_idx = np.random.randint(len(V))
    num_iter = 0
    while True:
        num_iter += 1
        newV = V.copy()
        iter_diff = 0
        for state in smdp.states():
            if smdp.isEnd(state):
                newV[state_to_idx_map[state]] = 0
            else:
                actions = smdp.actions(state)
                newV[state_to_idx_map[state]] = max(Q_vectorized(state, action,V) for action in actions)
        
        #min_diff = np.min(newV)
        #newV -= min_diff
        iter_diff = np.max(np.abs(V-newV))
        V = newV
        if num_iter >= max_iter or iter_diff < 1e-20:
            break
        #else:
        #    print(iter_diff)
        #if num_iter == 1:
        #    print('finished first iter!')
    #if num_iter >= max_iter:
    #    print('reached max num iterations!')
    #else:
    #    print('converged!')
    
    pi = {}
    resV = {}
    for state in smdp.states():
        resV[state] = V[state_to_idx_map[state]]
        if smdp.isEnd(state):
            pi[state] = 'none'
        else:
            actions = smdp.actions(state)
            pi[state] = actions[np.argmax([Q_vectorized(state,action,V) for action in actions])]
    return resV, pi

def valueIteration(smdp,max_iter=float('inf')):
    V = {}
    for state in smdp.states():
        V[state] = 0
    
    def Q(state,action):
        return sum(prob*(reward + smdp.discount()*V[new_state]) \
            for new_state, prob, reward in smdp.succProbReward(state,action))
    
    num_iter = 0
    while True:
        num_iter += 1
        newV = {}
        for state in smdp.states():
            if smdp.isEnd(state):
                newV[state] = 0
            else:
                actions = smdp.actions(state)
                newV[state] = max(Q(state, action) for action in actions)
        iter_diff = max(np.abs(V[state] - newV[state]) for state in smdp.states())
        if num_iter >= max_iter or iter_diff < 1e-3:
            break
        #else:
        #    print(iter_diff)
        V = newV
    
    #if num_iter >= max_iter:
    #    print('reached max num iterations!')
    #else:
    #    print('converged!')
    
    pi = {}
    for state in smdp.states():
        if smdp.isEnd(state):
            pi[state] = 'none'
        else:
            actions = smdp.actions(state)
            pi[state] = actions[np.argmax([Q(state,action) for action in actions])]
    return V, pi

def policyEvaluation_vectorized(smdp,pi,V={},max_iter=float('inf'),return_vector=False):
    idx_to_state_map = {}
    state_to_idx_map = {}
    
    V_vector = []
    count = 0
    for state in smdp.states():
        idx_to_state_map[count] = state
        state_to_idx_map[state] = count
        if not type(V).__module__ == np.__name__ and state in V:
            V_vector.append(V[state])
        elif type(V).__module__ == np.__name__:
            V_vector.append(V[count])
        else:
            V_vector.append(0)
        count += 1
    
    V_vector = np.array(V_vector,dtype=np.float64)

    def Q_vectorized(state,action):
        
        successor_list, prob_list, reward_list = smdp.succProbReward_vectorized(state,action)
        return np.sum(prob_list*(reward_list + smdp.discount()*V_vector))
    
    num_iter = 0
    while True:
        num_iter += 1
        newV = V_vector.copy()
        iter_diff = 0
        for state in smdp.states():
            if smdp.isEnd(state):
                newV[state_to_idx_map[state]] = 0
            else:
                newV[state_to_idx_map[state]] = Q_vectorized(state, pi[state])
            iter_diff = max([iter_diff,np.abs(V_vector[state_to_idx_map[state]] - newV[state_to_idx_map[state]])])
        if num_iter >= max_iter or iter_diff < 1e-3:
            break
        #else:
        #    print(iter_diff)
        V_vector = newV
    
    #if num_iter >= max_iter:
    #    print('reached max num iterations!')
    #else:
    #    print('converged!')
    
    if not return_vector:
        for state in smdp.states():
            V[state] = newV[state_to_idx_map[state]]
        return V
    else:
        return newV

def policyEvaluation(smdp,pi,V={},max_iter=float('inf')):
    if not V:
        for state in smdp.states():
            V[state] = 0
    
    def Q(state,action):
        return sum(prob*(reward + smdp.discount()*V[new_state]) \
            for new_state, prob, reward in smdp.succProbReward(state,action))
    
    num_iter = 0
    while True:
        num_iter += 1
        newV = {}
        iter_diff = 0
        for state in smdp.states():
            if smdp.isEnd(state):
                newV[state] = 0
            else:
                newV[state] = Q(state, pi[state])
            iter_diff = max([iter_diff,np.abs(V[state] - newV[state])])
        if num_iter >= max_iter or iter_diff < 1e-3:
            break
        else:
            print(iter_diff)
        V = newV
    
    if num_iter >= max_iter:
        print('reached max num iterations!')
    else:
        print('converged!')
    
    return V

def policyIteration(smdp,max_iter=float('inf')):
    
    print('running 1 iteration of value iteration for initial V and pi')
    V, pi = valueIteration(smdp,max_iter=1)
    
    def Q(state,action):
        return sum(prob*(reward + smdp.discount()*V[new_state]) \
            for new_state, prob, reward in smdp.succProbReward(state,action))

    policy_stable = False
    while not policy_stable:
        print('running policy eval')
        V = policyEvaluation(smdp,pi,V,max_iter=max_iter)
    
        policy_stable = True
        for state in smdp.states():
            prev_action = pi[state]
            if smdp.isEnd(state):
                pi[state] = 'none'
            else:
                actions = smdp.actions(state)
                pi[state] = actions[np.argmax([Q(state,action) for action in actions])]
            if prev_action != pi[state]:
                policy_stable = False
    return V, pi

def policyIteration_vectorized(smdp,max_iter=float('inf')):
    
    print('running 1 iteration of value iteration for initial V and pi')
    V_dict, pi = valueIteration_vectorized(smdp,max_iter=1)
    
    V = []
    count = 0
    idx_to_state_map = {}
    state_to_idx_map = {}
    for state in smdp.states():
        idx_to_state_map[count] = state
        state_to_idx_map[state] = count
        count += 1
        if state in V_dict:
            V.append(V_dict[state])
        else:
            V.append(0)
    
    V = np.array(V,dtype=np.float64)
    
    def Q_vectorized(state,action):
        successor_list, prob_list, reward_list = smdp.succProbReward_vectorized(state,action)
        return np.sum(prob_list*(reward_list + smdp.discount()*V))

    policy_stable = False
    while not policy_stable:
        print('running policy eval')
        V = policyEvaluation_vectorized(smdp,pi,V,max_iter=max_iter,return_vector=True)
    
        policy_stable = True
        for state in smdp.states():
            prev_action = pi[state]
            if smdp.isEnd(state):
                pi[state] = 'none'
            else:
                actions = smdp.actions(state)
                pi[state] = actions[np.argmax([Q_vectorized(state,action) for action in actions])]
            if prev_action != pi[state]:
                policy_stable = False
    
    retV_dict = {}
    for state in smdp.states():
        retV_dict[state] = V[state_to_idx_map[state]]
    
    return retV_dict, V, pi

def is_failure(pi,smdp,state):
    if smdp.get_queue_size(state) > 0 and pi[state] == 'none':
        return True
    elif smdp.is_SLO_violated(state,pi[state]):
        return True
    else:
        return False

def get_F_vec(pi,smdp,max_iter=100,initial_dist=None):
    p_f = get_fail_mat(pi,smdp)
    p_s = compute_p_s(smdp,pi,num_iters=max_iter,scale_by_qsize=True,initial_dist=initial_dist)
    F_vec = p_s*p_f
    return F_vec

def get_rewards(pi,smdp):
    states = smdp.states()
    rewards = np.zeros((len(states)))
    for i in range(len(states)):
        state = states[i]
        if pi[state] != 'none':
            rewards[i] = smdp.reward(state,pi[state])
    return rewards

def get_fail_mat(pi,smdp,scale_by_qsize=False):
    fail_mat = []
    for state in smdp.states():
        if not is_failure(pi,smdp,state):
            fail_mat.append(0.)
        elif not scale_by_qsize:
            fail_mat.append(1.)
        else:
            #fail_mat.append(smdp.get_queue_size(state))
            fail_mat.append(smdp.get_num_served(state,pi[state]))
    fail_mat = np.array(fail_mat)
    return fail_mat

def get_failure_prob(pi,smdp):
    fail_mat = get_fail_mat(pi,smdp)
    
    res = []
    for state in smdp.states():
        if not is_failure(pi,smdp,state):
            action = pi[state]
            successor_list, prob_list, reward_list = smdp.succProbReward_vectorized(state,action)
            res.append(np.sum(prob_list*fail_mat))
        else:
            res.append(1.)
    return np.array(res)

def get_prob_mat(pi,smdp):
    prob_mat = []
    for state_idx in range(len(smdp.states())):
        state = smdp.states()[state_idx]
        action = pi[state]
        if action != 'none':
            successor_list, prob_list, reward_list = smdp.succProbReward_vectorized(state,action)
            prob_mat.append(prob_list)
        else:
            cur_p = np.zeros(len(smdp.states()))
            cur_p[state_idx] = 1.
            prob_mat.append(cur_p)
    if scipy.sparse.issparse(prob_list):
        prob_mat = scipy.sparse.vstack(prob_mat).transpose().tocsr()
    else:
        prob_mat = np.array(prob_mat).T
    return prob_mat

def get_all_prob_mat(smdp):
    prob_mat = []
    for state in smdp.states():
        for action in smdp.actions(state):
            successor_list, prob_list, reward_list = smdp.succProbReward_vectorized(state,action)
            prob_mat.append(prob_list)
    if scipy.sparse.issparse(prob_list):
        prob_mat = scipy.sparse.vstack(prob_mat).transpose().tocsr()
    else:
        prob_mat = np.array(prob_mat).T
    return prob_mat

def compute_p_s(smdp,pi,num_iters=100,scale_by_qsize=False,initial_dist=None):
    prob_mat = get_prob_mat(pi,smdp)
    if initial_dist is None:
        #p_s = np.ones((len(smdp.states())))/len(smdp.states())
        p_s = np.zeros((len(smdp.states())))
        p_s[0] = 1.
    else:
        p_s = initial_dist.copy()

    for i in range(num_iters):
        p_s = prob_mat.dot(p_s)

    if scale_by_qsize:
        for i in range(len(smdp.states())):
            if pi[smdp.states()[i]] == 'none':
                p_s[i] = p_s[i] * smdp.get_queue_size(smdp.states()[i]) #assume everything is served if none
            else:
                p_s[i] = p_s[i] * smdp.get_num_served(smdp.states()[i],pi[smdp.states()[i]])
        p_normalization = np.sum(p_s)
        assert p_normalization >= 0, p_normalization
        if p_normalization > 0:
            p_s = p_s/np.sum(p_s)
    
    return p_s

def get_active_p_s(smdp,pi,num_iters=100,p_s=None):
    if p_s is None:
        p_s = compute_p_s(smdp,pi,num_iters)
    active_p_s = np.zeros_like(p_s)
    
    i = 0
    for state in smdp.states():
        action = pi[state]
        if action != 'none' and smdp.is_model_selected(action) and not is_failure(pi,smdp,state):
            active_p_s[i] = p_s[i]
        i += 1
    
    normalization = np.sum(active_p_s)
    assert normalization >= 0
    if normalization > 0:
        active_p_s = active_p_s/normalization
    return active_p_s

def compute_total_value(smdp,pi,num_iters=100):
    p_s = compute_p_s(smdp,pi,num_iters=num_iters,scale_by_qsize=True)
    active_p_s = get_active_p_s(smdp,pi,num_iters,p_s=p_s)
    
    reward_list = []
    for state in smdp.states():
        action = pi[state]
        if not is_failure(pi,smdp,state) and smdp.is_model_selected(action):
            model_selected = smdp.get_model_selected(action)
            reward_list.append(smdp.model_accuracy[model_selected])
            #reward_list.append(smdp.reward(state,action))
        else:
            reward_list.append(0)
    reward_list = np.array(reward_list)
    
    total_V = np.sum(active_p_s*reward_list)
    return total_V

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def input_pi(name):
    f = open(name+'_pi.json','r')
    read_dict = json.load(f)
    out_dict = dict()
    f.close()
    for k,v in read_dict.items():
        if ',' in k:
            state = tuple([int(entry) for entry in k[1:-1].split(',')])
        else:
            state = int(k)
        if not isinstance(v,int):
            val = tuple(v)
        else:
            val = v
        out_dict[state] = val
    return out_dict
    
def output_pi(name,pi):
    out_dict = dict()
    for k,v in pi.items():
        if v != 'none' and not isinstance(v,tuple):
            out_dict[str(k)] = int(v)
        else:
            out_dict[str(k)] = v
    f = open(name+'_pi.json','w')
    json.dump(out_dict,f,cls=NpEncoder)
    f.close()

def get_exp_fname(exp_dir,qrate,target_violation,SLO,num_inst):
    slo_str = str(int(SLO/1000))
    exp_name = os.path.join(exp_dir+"_instances-"+str(num_inst)+'_'+slo_str+'ms_'+str(target_violation),"test_rate-"+str(qrate))
    return exp_name
    
def output_V(name,V):
    out_dict = dict()
    for k,v in V.items():
        out_dict[str(k)] = v
    f = open(name+'_V.json','w')
    json.dump(out_dict,f,cls=NpEncoder)
    f.close()

def input_V(name):
    f = open(name+'_V.json','r')
    read_dict = json.load(f)
    out_dict = dict()
    f.close()
    for k,v in read_dict.items():
        try:
            out_dict[ast.literal_eval(k)] = v
        except:
            out_dict[k] = v
    return out_dict
    
def policyIteration_worker(args):
    name,smdp,num_iters,target_violation = args
    V_dict, V, pi = policyIteration_vectorized(smdp,max_iter=num_iters)
    new_pi = pi.copy()
    #new_pi = get_safe_policy_v6(pi,smdp,max_iter=100,prob_constraint=target_violation)
    safety_V_dict = safe_policyEval(new_pi,smdp,max_iter=100,return_V=True)
    safety_V_dict[-1] = np.sum(get_F_vec(new_pi,smdp,max_iter=100))
    V_dict = policyEvaluation_vectorized(smdp,new_pi,max_iter=10)
    total_val = compute_total_value(smdp,new_pi)
    V_dict[-1] = total_val
    
    os.makedirs(os.path.dirname(name),exist_ok=True)
    output_pi(name,new_pi)
    output_V(name+'_safety',safety_V_dict)
    output_V(name,V_dict)
    print('done!')
    return new_pi

def myvalueiteration_worker(args):
    start_t = time.time()
    name,smdp,num_iters,target_violation = args
    try:
        print(name,'start')
        V_dict, new_pi = my_valueIteration_vectorized(smdp,max_iter=num_iters)
        #print(name,'completed!')
        #safety_V_dict = safe_policyEval(new_pi,smdp,max_iter=10,return_V=True)
        
        #cmp_pi = get_optimistic_policy(smdp)
        #cmp_value = compute_total_value(smdp,cmp_pi,num_iters=100)
        #cmp_safety = np.sum(get_F_vec(cmp_pi,smdp,max_iter=100))
        
        #if cmp_pi != new_pi:
        #    print("different from optimistic")
        #else:
        #    print("same as optimistic!")
        f_vec = get_F_vec(new_pi,smdp,max_iter=1000)
        safety_dict = {-1:np.sum(f_vec)}
        
        V_dict = {-1:compute_total_value(smdp,new_pi,num_iters=1000)}
        #print('value:',V_dict,'safety:',safety_dict,'cmp value',cmp_value,'cmp safety',cmp_safety)
        #print('num_inst:',smdp.num_inst,'rate:',smdp.rate*1e6,'SLO:',smdp.SLO,'value:',V_dict,'safety:',safety_dict,'end prob:',f_vec[-1])
        os.makedirs(os.path.dirname(name),exist_ok=True)
        output_pi(name,new_pi)
        #output_V(name+'_safety',safety_V_dict)
        output_V(name+'_policysafety',safety_dict)
        output_V(name,V_dict)
        print(name,'done!')
        smdp.clear_cache()
        return new_pi, smdp, time.time()-start_t
    except Exception as e:
        print('error happened!',name)
        raise e
        
def get_optimistic_policy(smdp):
    pi = {}
    for state in smdp.states():
        actions = smdp.actions(state)
        if len(actions) > 0:
            pi[state] = actions[-1]
        else:
            pi[state] = 'none'
    return pi

def get_minimum_policy(smdp):
    pi = {}
    for state in smdp.states():
        if smdp.isEnd(state):
            pi[state] = 'none'
        elif smdp.get_queue_size(state) > 0:
            pi[state] = 1
        else:
            pi[state] = 0
    return pi

def valueIteration_worker(args):
    start_t = time.time()
    name,smdp,num_iters,target_violation = args
    try:
        print(name,'start')
        V_dict, new_pi = valueIteration_vectorized(smdp,max_iter=num_iters)
        #new_pi = input_pi(name)
        
        #new_pi = get_safe_policy_v6(new_pi,smdp,max_iter=10,prob_constraint=0.01)
        #print(name,'completed!')
        #safety_V_dict = safe_policyEval(new_pi,smdp,max_iter=10,return_V=True)
        
        #cmp_pi = get_optimistic_policy(smdp)
        #cmp_value = compute_total_value(smdp,cmp_pi,num_iters=100)
        #cmp_safety = np.sum(get_F_vec(cmp_pi,smdp,max_iter=100))
        
        #if cmp_pi != new_pi:
        #    print("different from optimistic")
        #else:
        #    print("same as optimistic!")
        f_vec = get_F_vec(new_pi,smdp,max_iter=1000)
        safety_dict = {-1:np.sum(f_vec)}
        
        V_dict = {-1:compute_total_value(smdp,new_pi,num_iters=1000)}
        #print('value:',V_dict,'safety:',safety_dict,'end prob:',f_vec[-1],'cmp value',cmp_value,'cmp safety',cmp_safety)
        #print('value:',V_dict,'safety:',safety_dict,'end prob:',f_vec[-1])
        os.makedirs(os.path.dirname(name),exist_ok=True)
        output_pi(name,new_pi)
        #output_V(name+'_safety',safety_V_dict)
        output_V(name+'_policysafety',safety_dict)
        output_V(name,V_dict)
        print(name,'done!')
        smdp.clear_cache()
        return new_pi, smdp, time.time()-start_t
    except Exception as e:
        print('error happened!',name)
        raise e
    except:
        print('error happened!',name)

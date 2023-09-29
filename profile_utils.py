import sys
PROFILE_HOME='./profiles'
sys.path.append(PROFILE_HOME)
from scipy.stats import norm
import sklearn
from model_run import *
from seq_models import *
from image_models import *

def get_pt_latency_dist_accuracy_list(home_dir,models,max_batch_size=32,percentile=90):
    acc_list = []
    lat_list = []
    for model_name in models:
        assert model_name in image_accuracy_dict or model_name in seq_accuracy_dict
        if model_name in image_accuracy_dict:
            model_acc = image_accuracy_dict[model_name]
        else:
            model_acc = seq_accuracy_dict[model_name]
        acc_list.append(model_acc)
        lat_list.append([])
        cur_bsize = []
        for bsize in range(1,max_batch_size+1):
            try:
                profile = results_loader(get_save_str(dir_name=PROFILE_HOME+"/"+home_dir,model_name=model_name,num_models=1,batch_size=bsize))
                p99 = (1.0*np.percentile(profile,percentile))*1e6
                dist = [p99]
                lat_list[-1].append(dist)
                cur_bsize.append([bsize])
            except Exception as e:
                pass
        predictor = sklearn.linear_model.LinearRegression().fit(cur_bsize,lat_list[-1])
        cur_list = predictor.predict([[bsize] for bsize in range(1,max_batch_size+1)])
        lat_list[-1] = [ entry for entry in cur_list]
        for bsize in range(1,max_batch_size+1):
            try:
                profile = results_loader(get_save_str(dir_name=PROFILE_HOME+"/"+home_dir,model_name=model_name,num_models=1,batch_size=bsize))
                p99 = (1.0*np.percentile(profile,percentile))*1e6
                dist = [p99]
                lat_list[-1][bsize-1] = dist
            except Exception as e:
                pass
        for i in range(len(lat_list[-1])-1):
            if lat_list[-1][len(lat_list[-1])-1-i][0] < lat_list[-1][len(lat_list[-1])-1-i-1][0]:
                lat_list[-1][len(lat_list[-1])-1-i-1] = lat_list[-1][len(lat_list[-1])-1-i]
    mass_list = np.ones((len(models),max_batch_size,1))
    return np.array(lat_list), np.array(mass_list), np.array(acc_list)/100


def get_sim_gaussian(mean,std_dev_frac,dist_size=2,max_quantile=0.99):
    res = []
    probmass = []
    for i in range(dist_size):
        p = (i+1)*max_quantile/dist_size
        res.append(norm.ppf(p, loc=mean, scale=std_dev_frac*mean))
        if i == 0:
            probmass.append(norm.cdf(res[-1], loc=mean,scale=std_dev_frac*mean))
        else:
            probmass.append(norm.cdf(res[-1], loc=mean,scale=std_dev_frac*mean) - norm.cdf(res[-2], loc=mean,scale=std_dev_frac*mean))
    probmass[-1] += 1 - np.sum(probmass)
    return np.array(res), np.array(probmass)

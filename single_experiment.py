
### execute this function to train and test the vae-model

from vaemodel import Model
import numpy as np
import pickle
import torch
import os
import heapq
import random


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     

def printHyperparameters(hyperparameters):
    print("##################The key parameters are as follows:##################\n")
    print("generalized: ", hyperparameters['generalized'])
    print("seen/unseen split: ", hyperparameters['ratio_mode'][0])
    print("img_feature_file_path: ", hyperparameters['img_feature_file_path'])
    print("att_feature_file_path: ", hyperparameters['att_feature_file_path'])
    print("label_file_path: ", hyperparameters['label_file_path'])
    print("cross_reconstruction_factor: ", hyperparameters['cross_reconstruction_factor'])
    print("distribution_alignment_factor: ", hyperparameters['distribution_alignment_factor'])
    print("loss_v2v_factor: ", hyperparameters['loss_v2v_factor'])
    print("loss_v2s_factor: ", hyperparameters['loss_v2s_factor'])
    print("loss_s2v_factor: ", hyperparameters['loss_s2v_factor'])
    print("loss_s2v_factor: ", hyperparameters['loss_s2v_factor'])
    print("c_way: ", hyperparameters['c_way'])
    print("k_shot: ", hyperparameters['k_shot'])
    print("tau: ", hyperparameters['tau'])
    print("latent_size: ", hyperparameters['latent_size'])



########################################
# The hyperparameters
########################################
hyperparameters = {
    'device': 'cuda',   
    'lr_gen_model': 0.00015, 
    'generalized': False,
    'epochs': 51,
    'loss': 'l1', 
    'auxiliary_data_source' : 'attributes',
    'lr_cls': 0.001,
    'dataset': 'RSdataset',
    'hidden_size_rule': {'resnet_features': (512, 512),
                        'attributes': (256, 256) },
    'latent_size':64, 
    
    'img_feature_file_path' : './data/ours/rest18_tune.csv',
    'att_feature_file_path' : './data/ours/bert.mat',
    'label_file_path' : './data/ours/label_rest18_tune.txt',
    
    'cross_reconstruction_factor' : 10, 
    'distribution_alignment_factor' : 1, 
    'loss_v2v_factor' : 100, 
    'loss_v2s_factor' : 100, 
    'loss_s2v_factor' : 10, 
    
    'c_way' : 5, 
    'k_shot' : 5,
    'tau' : 2, 
    
    'num_classes' : 70,
    'ratio_mode' : ('4030', 30) # ('5020',20), ('4030',30)
}



cls_train_steps = [
      {'dataset': 'RSdataset',  'generalized': True, 'cls_train_steps': 23},
      {'dataset': 'RSdataset',  'generalized': False, 'cls_train_steps': 50}
      ]
      
 
hyperparameters['cls_train_steps'] = [x['cls_train_steps']  for x in cls_train_steps
                                        if all([hyperparameters['dataset']==x['dataset'],
                                        hyperparameters['generalized']==x['generalized'] ])][0]


if hyperparameters['generalized']:
    hyperparameters['samples_per_class'] = {'RSdataset': (200, 400)}
else:
    hyperparameters['samples_per_class'] = {'RSdataset': (0, 200)}
    

if hyperparameters['generalized']:
    hyperparameters['samples_per_class_in_batch'] = {'RSdataset': 600}
else:
    hyperparameters['samples_per_class_in_batch'] = {'RSdataset': 800}


if hyperparameters['generalized']:
    results_log_path= "./data/results/gzsl_"+str(hyperparameters['ratio_mode'][0])+".txt"
else:
    results_log_path= "./data/results/zsl_"+str(hyperparameters['ratio_mode'][0])+".txt"





########################################
# The main
########################################
printHyperparameters(hyperparameters)

results_log_file =open(results_log_path,'w')
best_result_parameter = 0.0
for j in range(1,6):
    setup_seed(10)
    
    print("############################The split file:"+str(j)+"#########################\n")
    results_log_file.write("############################The split file:"+str(j)+"#########################\n")    
    
    model = Model( hyperparameters, j, results_log_file)
    model.to(hyperparameters['device'])
    losses, best_result, best_epoch = model.train_vae()
        
    print("***The best result under the split file of "+str(j)+": \n")
    print("***Best result: "+str(best_result)+" at the epoch:"+str(best_epoch)+"\n")
    results_log_file.write("***The best result under the split file of "+str(j)+": \n")
    results_log_file.write("***Best result: "+str(best_result)+" at the epoch:"+str(best_epoch)+"\n")
        
    
results_log_file.close()



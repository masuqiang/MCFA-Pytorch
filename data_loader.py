

import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing
import sys
import os
from pathlib import Path
import pickle
import copy
import csv
import random
import matplotlib.pyplot as plt
from sklearn import manifold
from matplotlib import cm


             
def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i

    return mapped_label

class DATA_LOADER(object):
    def __init__(self, hyperparameters, iteration_num, generalized, ratio_mode):

        self.img_feature_file_path = hyperparameters['img_feature_file_path']
        self.att_feature_file_path = hyperparameters['att_feature_file_path']
        self.label_file_path = hyperparameters['label_file_path']
        self.device = hyperparameters['device']
        
        self.auxiliary_data_source = hyperparameters['auxiliary_data_source']
        
        self.iteration_num = iteration_num
        
        self.generalized = generalized
        
        self.ratio_mode = ratio_mode
        
        self.read_matdataset()


    
    def next_batch(self, c, k, n):
        #####################################################################
        # gets batch from train_feature = 7057 samples from 150 train classes
        #####################################################################
        selected_class_index = random.sample(range(0, len(self.train_labels_unique)), c)
        selected_sample_index_per_class = random.sample(range(0, n), k)
        selected_class = self.train_labels_unique[selected_class_index]
        
        selected_sample_index = []
        for class_id in selected_class:
            for random_sample_index in selected_sample_index_per_class:
                sample_index = self.train_class_start_dic[class_id.item()]+random_sample_index
                selected_sample_index.append(sample_index)
        
        batch_feature = self.train_img_features[selected_sample_index].to(self.device)
        batch_label = self.train_labels[selected_sample_index]
        batch_label_att = self.aux_data[selected_class].to(self.device)
        batch_label = map_label(batch_label, selected_class).to(self.device)
        batch_feature.requires_grad = False
        batch_label_att.requires_grad = False
        return batch_label, batch_feature, batch_label_att
    
       
        
    def text_read(self, filename):
        try:
            file = open(filename,'r')
        except IOError:
            error = []
            return error
        content = file.readlines()
        for i in range(len(content)):
            content[i] = content[i][:len(content[i])-1]
        file.close()
        return content


    def read_matdataset(self):
    
        feature = []
        with open(self.img_feature_file_path, "r") as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                feature.append(line)
        feature = np.array(feature).astype(float)  
       
        
        if len(self.label_file_path)>0:
            label = self.text_read(self.label_file_path)
            label = np.array(label).squeeze().astype(int) - 1
            print("label shape:", label.shape)
            
            
        
        if self.generalized == True:
            file_dir = "./data/ours/gzsl_"+str(self.ratio_mode)+"_5split/"
        else:
            file_dir = "./data/ours/zsl_"+str(self.ratio_mode)+"_5split/"
        
         
        seen_file_name_str = file_dir + "seen_class_loc_"+ str(self.ratio_mode)+"_" +str(self.iteration_num) + ".txt"
        seen_class_index = self.text_read(seen_file_name_str)
        seen_class_index = np.array(seen_class_index).squeeze().astype(int)
        unseen_file_name_str = file_dir + "unseen_class_loc_"+ str(self.ratio_mode)+"_"+str(self.iteration_num) + ".txt" 
        unseen_class_index = self.text_read(unseen_file_name_str)
        unseen_class_index = np.array(unseen_class_index).squeeze().astype(int)         
        trainval_loc = seen_class_index
        test_unseen_loc = unseen_class_index
        
        if self.generalized == True:
            test_seen_file_name_str = file_dir + "test_seen_class_loc_"+ str(self.ratio_mode)+"_"+str(self.iteration_num) + ".txt" 
            test_seen_index = self.text_read(test_seen_file_name_str)
            test_seen_loc =np.array(test_seen_index).squeeze().astype(int)
       
        w2v = []
        data = sio.loadmat(self.att_feature_file_path)
        for i in data:
            w2v.append(data[i])
        new_w2v = w2v[3:]  
        sem_feats = np.array(new_w2v).squeeze()  #
        self.aux_data = torch.from_numpy(sem_feats).float()

        
        scaler = preprocessing.MinMaxScaler()
        train_feature = scaler.fit_transform(feature[trainval_loc])
        if self.generalized == True:
            test_seen_feature = scaler.transform(feature[test_seen_loc])
        test_unseen_feature = scaler.transform(feature[test_unseen_loc])
        
        self.train_img_features = torch.from_numpy(train_feature).float().to(self.device)
        self.train_labels = torch.from_numpy(label[trainval_loc]).long().to(self.device)
        self.ntrain = self.train_img_features.size()[0]
        self.train_labels_unique = torch.from_numpy(np.unique(self.train_labels.cpu().numpy()))
        
        
        self.train_class_start_dic = {}
        pre_label = -1
        for i in range(len(self.train_labels)):
             if self.train_labels[i] != pre_label:
                 self.train_class_start_dic[self.train_labels[i].item()] = i
             pre_label = self.train_labels[i]
        
        self.test_img_features = torch.from_numpy(test_unseen_feature).float().to(self.device)
        self.test_labels = torch.from_numpy(label[test_unseen_loc]).long().to(self.device)
        
        
        if self.generalized == True:
            test_seen_feature = torch.from_numpy(test_seen_feature).float().to(self.device)
            test_seen_label = torch.from_numpy(label[test_seen_loc]).long().to(self.device)
        
        self.seenclasses = torch.from_numpy(np.unique(self.train_labels.cpu().numpy())).to(self.device)
        self.novelclasses = torch.from_numpy(np.unique(self.test_labels.cpu().numpy())).to(self.device)
        
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.novelclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class+self.ntest_class).long()
       

        
        self.data = {}
        self.data['train_seen'] = {}
        self.data['train_seen']['resnet_features'] = self.train_img_features
        self.data['train_seen']['labels']= self.train_labels
        self.data['train_seen'][self.auxiliary_data_source] = self.aux_data[self.train_labels].to(self.device)

        if self.generalized == True:
            self.data['test_seen'] = {}
            self.data['test_seen']['resnet_features'] = test_seen_feature
            self.data['test_seen']['labels'] = test_seen_label

        self.data['test_unseen'] = {}
        self.data['test_unseen']['resnet_features'] = self.test_img_features
        self.data['test_unseen'][self.auxiliary_data_source] = self.aux_data[self.test_labels].to(self.device)
        self.data['test_unseen']['labels'] = self.test_labels

        self.novelclass_aux_data = self.aux_data[self.novelclasses].to(self.device)
        self.seenclass_aux_data = self.aux_data[self.seenclasses].to(self.device)
        
        

def text_read(filename):
    try:
        file = open(filename,'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
    for i in range(len(content)):
        content[i] = content[i][:len(content[i])-1]
    file.close()
    return content







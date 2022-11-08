import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F



###############################################################################################################
##按照supcon的方式来计算对比损失
###############################################################################################################    
        
class Inter_Mode_Loss(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(Inter_Mode_Loss,self).__init__()
        
        self.temperature = temperature
          
    #如下损失函数是按照simCLR对比损失方式新构建的    
    def forward(self,feat1,label):
        #print("**********in inter_mode_loss**********")
        # 将输入的特征矩阵进行l2归一化，方便下面的內积运算
        feat1 = F.normalize(feat1, dim=-1)
        #每个样本与batch中其他样本的內积相似性
        sim_matrix = torch.exp(torch.mm(feat1, feat1.t().contiguous()) / self.temperature)
        #去掉对角线上的值
        mask = (torch.ones_like(sim_matrix) - torch.eye(feat1.shape[0], device=sim_matrix.device)).bool()
        #构造分母
        negative_sim = sim_matrix.masked_select(mask).view(feat1.shape[0], -1)
        negative_sim = negative_sim.sum(dim=-1)

        #构造分子
        mask = label.repeat(feat1.shape[0], 1)
        mask2 = mask.transpose(1, 0)
        mask = mask2 - mask
        mask[mask!=0] = -1
        mask[mask == 0] = 1
        mask[mask<0] = 0
        mask = (mask - torch.eye(feat1.shape[0], device=sim_matrix.device)).bool()
        pos_sim = sim_matrix.masked_select(mask).view(feat1.shape[0], -1).mean(dim=-1)

        #return (- torch.log(pos_sim / negative_sim)).mean()
        return torch.sum(- torch.log(pos_sim / negative_sim))



class Cross_Mode_Loss(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(Cross_Mode_Loss,self).__init__()
        
        self.temperature = temperature

          
    def forward(self,feature_from_img, feature_from_att, label):
        #print("**********in cross_model_loss**********")from Model.vit import Transformer
        feature_from_img = F.normalize(feature_from_img, dim=-1)
        feature_from_att = F.normalize(feature_from_att, dim=-1)
        
        
        #total_loss = 0.0
        sim_matrix = torch.exp(torch.mm(feature_from_img, feature_from_att.t().contiguous())/self.temperature)
        
        
        #计算每个视觉特征与自己所在的类别语义特在的相识度
        pos_sim = torch.exp(torch.sum(feature_from_img*feature_from_att[label], dim=-1)/self.temperature)
        #cross_loss_vtos = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        cross_loss_vtos = torch.sum(- torch.log(pos_sim / sim_matrix.sum(dim=-1)))
        
        #return cross_loss_vtos
        
        
        #计算每个语义与自己所在的类别的样本的相识度
        mask = torch.unique(label).repeat(feature_from_img.shape[0],1).transpose(1, 0)
        mask2 = label.repeat(feature_from_att.shape[0], 1)
        mask = mask - mask2
        mask[mask != 0] = -1
        mask[mask == 0] = 1
        mask[mask < 0] = 0
        pos_sim = sim_matrix.t().masked_select(mask.bool()).view(feature_from_att.shape[0], -1).mean(dim=-1)
        #cross_loss_stov = (- torch.log(pos_sim / sim_matrix.t().sum(dim=-1))).mean()
        cross_loss_stov = torch.sum(- torch.log(pos_sim / sim_matrix.t().sum(dim=-1)))
        
        #return cross_loss_stov
        return cross_loss_vtos, cross_loss_stov 
       
        
        
 
 

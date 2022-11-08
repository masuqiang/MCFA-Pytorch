#vaemodel
import copy
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.utils import data
from data_loader import DATA_LOADER as dataloader
import final_classifier as  classifier
import models
from torch.autograd import Variable
from ContrastiveLoss_new import Inter_Mode_Loss,Cross_Mode_Loss




class LINEAR_LOGSOFTMAX(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX, self).__init__()
        self.fc = nn.Linear(input_dim,nclass)
        self.logic = nn.LogSoftmax(dim=1)
        self.lossfunction =  nn.NLLLoss()

    def forward(self, x):
        o = self.logic(self.fc(x))
        return o

class Model(nn.Module):

    def __init__(self,hyperparameters, iteration_num, results_log_file):
        super(Model,self).__init__()
       
        ############################################################################################
        # 初始化函数中的参数设置部分
        #############################################################################################
        self.device = hyperparameters['device']
        self.auxiliary_data_source = hyperparameters['auxiliary_data_source']
        self.all_data_sources  = ['resnet_features',self.auxiliary_data_source]
        self.DATASET = hyperparameters['dataset']
        self.latent_size = hyperparameters['latent_size']
        self.hidden_size_rule = hyperparameters['hidden_size_rule']
        self.generalized = hyperparameters['generalized']
        self.classifier_batch_size = 32
        self.img_seen_samples   = hyperparameters['samples_per_class'][self.DATASET][0]
        self.att_unseen_samples = hyperparameters['samples_per_class'][self.DATASET][1]
        self.reco_loss_function = hyperparameters['loss']
        self.nepoch = hyperparameters['epochs']
        self.lr_cls = hyperparameters['lr_cls']
        self.cls_train_epochs = hyperparameters['cls_train_steps']
        self.ratio_mode = hyperparameters['ratio_mode'][0]
        
        self.iteration_num = iteration_num
        self.results_log_file = results_log_file
        self.best_result = 0.0
        self.best_epoch = 0
        
        
        self.img_feature_file_path = hyperparameters['img_feature_file_path']
        self.att_feature_file_path = hyperparameters['att_feature_file_path']
        self.label_file_path = hyperparameters['label_file_path']
        self.dataset = dataloader(hyperparameters, self.iteration_num, self.generalized, self.ratio_mode)
        
      
        self.c_way = hyperparameters['c_way']
        self.k_shot = hyperparameters['k_shot']
        self.tau = hyperparameters['tau']
        self.n_per_class = hyperparameters['samples_per_class_in_batch'][self.DATASET]
        
        self.cross_reconstruction_factor = hyperparameters['cross_reconstruction_factor']
        self.distribution_alignment_factor = hyperparameters['distribution_alignment_factor']
        self.loss_v2v_factor = hyperparameters['loss_v2v_factor']
        self.loss_v2s_factor = hyperparameters['loss_v2s_factor']
        self.loss_s2v_factor = hyperparameters['loss_s2v_factor']
        
       
       
        self.num_classes=hyperparameters['num_classes']
        self.num_novel_classes =  hyperparameters['ratio_mode'][1] 
        
        #######################################################################################################################
        # 初始化函数中的模型定义部分
        #######################################################################################################################
        #feature_dimensions = [2048, self.dataset.aux_data.size(1)]
        feature_dimensions = [512, self.dataset.aux_data.size(1)]
  
        self.encoder = {}
        for datatype, dim in zip(self.all_data_sources,feature_dimensions):
            self.encoder[datatype] = models.encoder_template(dim,self.latent_size,self.hidden_size_rule[datatype],self.device)
            print(str(datatype) + ' ' + str(dim))

        self.decoder = {}
        for datatype, dim in zip(self.all_data_sources,feature_dimensions):
            self.decoder[datatype] = models.decoder_template(self.latent_size,dim,self.hidden_size_rule[datatype],self.device)

        parameters_to_optimize = list(self.parameters())
        for datatype in self.all_data_sources:
            parameters_to_optimize +=  list(self.encoder[datatype].parameters())
            parameters_to_optimize +=  list(self.decoder[datatype].parameters())
        self.optimizer  = optim.Adam( parameters_to_optimize ,lr=hyperparameters['lr_gen_model'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
        
        
        if self.reco_loss_function=='l2':
            self.reconstruction_criterion = nn.MSELoss(size_average=False)
        elif self.reco_loss_function=='l1':
            self.reconstruction_criterion = nn.L1Loss(size_average=False)#True, False
        
      
        #contrastive loss
        self.inter_mode_loss = Inter_Mode_Loss(temperature=self.tau)
        self.cross_mode_loss = Cross_Mode_Loss(temperature=self.tau)
      
      
   
    def reparameterize(self, mu, logvar):
        if self.reparameterize_with_noise:
            sigma = torch.exp(logvar)
            eps = torch.cuda.FloatTensor(logvar.size()[0],1).normal_(0,1)
            eps  = eps.expand(sigma.size())
            return mu + sigma*eps
        else:
            return mu

    def forward(self):
        pass
        
  
    def map_label(self,label, classes):
        mapped_label = torch.LongTensor(label.size()).to(self.device)
        for i in range(classes.size(0)):
            mapped_label[label==classes[i]] = i
        return mapped_label
        
       
    def trainstep(self, img, label, att):
       
        ##############################################
        # Encode image features and additional
        # features
        ##############################################

        mu_img, logvar_img = self.encoder['resnet_features'](img)
        z_from_img = self.reparameterize(mu_img, logvar_img)

        mu_att, logvar_att = self.encoder[self.auxiliary_data_source](att)
        z_from_att = self.reparameterize(mu_att, logvar_att)

        ##############################################
        # Reconstruct inputs
        ##############################################

        img_from_img = self.decoder['resnet_features'](z_from_img)
        att_from_att = self.decoder[self.auxiliary_data_source](z_from_att)
        
        reconstruction_loss = self.reconstruction_criterion(img_from_img, img) \
                              + self.reconstruction_criterion(att_from_att, att)
        
        ##############################################
        # KL-Divergence
        ##############################################

        KLD = (0.5 * torch.sum(1 + logvar_att - mu_att.pow(2) - logvar_att.exp())) \
              + (0.5 * torch.sum(1 + logvar_img - mu_img.pow(2) - logvar_img.exp()))
                
        loss = reconstruction_loss - KLD
        
        
        ##############################################
        # Cross Reconstruction Loss
        ##############################################
        
        img_from_att = self.decoder['resnet_features'](z_from_att[label]) 
        att_from_img = self.decoder[self.auxiliary_data_source](z_from_img)

        cross_reconstruction_loss = self.reconstruction_criterion(img_from_att, img) \
                                    + self.reconstruction_criterion(att_from_img, att[label])
                                    
        cross_reconstruction_loss = self.cross_reconstruction_factor*cross_reconstruction_loss
        
        if cross_reconstruction_loss>0:
            loss += cross_reconstruction_loss
            
        
        ##############################################
        # Distribution Alignment
        ##############################################
        
        distance = torch.sqrt(torch.sum((mu_img - mu_att[label]) ** 2, dim=1) + \
                              torch.sum((torch.sqrt(logvar_img.exp()) - torch.sqrt(logvar_att[label].exp())) ** 2, dim=1))
        distribution_alignment_loss = distance.sum()
        
        distribution_alignment_loss = self.distribution_alignment_factor*distribution_alignment_loss
        
        if distribution_alignment_loss >0:
            loss += distribution_alignment_loss
        
        
        ##############################################
        # contrastive loss
        ##############################################
        
        loss_v2v = self.inter_mode_loss(z_from_img,label)
        loss_v2v = loss_v2v*self.loss_v2v_factor
        loss += loss_v2v
        
        
        #loss_v2s = self.cross_mode_loss(z_from_img, z_from_att,label)
        #loss_s2v = self.cross_mode_loss(z_from_img, z_from_att,label)
        loss_v2s, loss_s2v = self.cross_mode_loss(z_from_img, z_from_att,label)
        
        
        loss_v2s = loss_v2s*self.loss_v2s_factor
        loss += loss_v2s
        
        
        loss_s2v = loss_s2v*self.loss_s2v_factor
        loss += loss_s2v
        
        
        ##############################################
        # call the optimizer
        ##############################################
       
        self.optimizer.zero_grad()     
        loss.backward()
        self.optimizer.step()
        
        #return loss.item()
        return reconstruction_loss.item(),cross_reconstruction_loss.item(),distribution_alignment_loss.item(),loss_v2v.item(),loss_v2s.item(),loss_s2v.item(),loss.item()


    def train_vae(self):

        losses = []
        #leave both statements
        self.train()
        self.reparameterize_with_noise = True
        
        
        for epoch in range(1, self.nepoch ):
            self.current_epoch = epoch
      
            i=-1
            for iters in range(0, self.dataset.ntrain, self.c_way*self.k_shot):
                i+=1

                #print("Training VAE")
                label, img, att = self.dataset.next_batch(self.c_way, self.k_shot, self.n_per_class)
                #loss = self.trainstep(img, label, att)
                reconstruction_loss,cross_reconstruction_loss,distribution_alignment_loss,loss_v2v,loss_v2s,loss_s2v,loss = self.trainstep(img, label, att)

                if i%300==0:
                    print('epoch: ' + str(epoch) + '|iter: ' + str(i) + 
                     '|total_loss: ' +  str(loss)[:10] +'|reconstruction: ' +  str(reconstruction_loss)[:8] + '|cross_reconstruction: ' +  str(cross_reconstruction_loss)[:8] + '|distribution_alignment: ' +  str(distribution_alignment_loss)[:8]+'|v2v: ' +  str(loss_v2v)[:8] + '|v2s: ' +  str(loss_v2s)[:8]+ '|s2v: ' +  str(loss_s2v)[:8])
                    #print('epoch: ' + str(epoch) + '|iter: ' + str(i) + '|total_loss: ' +  str(loss)[:10])


                if i%300==0 and i>0:
                    losses.append(loss)
           
        # turn into evaluation mode:
        for key, value in self.encoder.items():
            self.encoder[key].eval()
        for key, value in self.decoder.items():
            self.decoder[key].eval()
        self.train_classifier()
        
        return losses, self.best_result, self.best_epoch

    def train_classifier(self, show_plots=False):
    
        cls_seenclasses = self.dataset.seenclasses
        cls_novelclasses = self.dataset.novelclasses
        
        train_seen_feat = self.dataset.data['train_seen']['resnet_features']
        train_seen_label = self.dataset.data['train_seen']['labels']
        
        novelclass_aux_data = self.dataset.novelclass_aux_data  
        seenclass_aux_data = self.dataset.seenclass_aux_data

        novel_corresponding_labels = self.dataset.novelclasses.long().to(self.device)
        seen_corresponding_labels = self.dataset.seenclasses.long().to(self.device)
       
        #data for testing
        novel_test_feat = self.dataset.data['test_unseen'][ 'resnet_features'] 
        test_novel_label = self.dataset.data['test_unseen']['labels']  
        if self.generalized == True:    
            seen_test_feat = self.dataset.data['test_seen']['resnet_features'] 
            test_seen_label = self.dataset.data['test_seen']['labels']  
        


        # in ZSL mode:
        if self.generalized == False:
            novel_corresponding_labels = self.map_label(novel_corresponding_labels, novel_corresponding_labels)
            test_novel_label = self.map_label(test_novel_label, cls_novelclasses)
            cls_novelclasses = self.map_label(cls_novelclasses, cls_novelclasses)


        if self.generalized:
            print('mode: gzsl')
            clf = LINEAR_LOGSOFTMAX(self.latent_size, self.num_classes)
        else:
            print('mode: zsl')
            print("self.num_novel_classes",self.num_novel_classes)
            clf = LINEAR_LOGSOFTMAX(self.latent_size, self.num_novel_classes)


        clf.apply(models.weights_init)

        with torch.no_grad():

            ####################################
            # preparing the test set
            # convert raw test data into z vectors
            ####################################

            self.reparameterize_with_noise = False
            
            mu1, var1 = self.encoder['resnet_features'](novel_test_feat)
            test_novel_X = self.reparameterize(mu1, var1).to(self.device).data
            test_novel_Y = test_novel_label.to(self.device)
            
            if self.generalized == True:
                mu2, var2 = self.encoder['resnet_features'](seen_test_feat)
                test_seen_X = self.reparameterize(mu2, var2).to(self.device).data
                test_seen_Y = test_seen_label.to(self.device)
            else:
                test_seen_X = torch.tensor([])
                test_seen_Y = torch.tensor([])
           
            
            ####################################
            # preparing the train set:
            # chose n random image features per
            # class. If n exceeds the number of
            # image features per class, duplicate
            # some. Next, convert them to
            # latent z features.
            ####################################

            self.reparameterize_with_noise = True

            def sample_train_data_on_sample_per_class_basis(features, label, sample_per_class):
                sample_per_class = int(sample_per_class)

                if sample_per_class != 0 and len(label) != 0:

                    classes = label.unique()

                    for i, s in enumerate(classes):

                        features_of_that_class = features[label == s, :]  # order of features and labels must coincide
                        # if number of selected features is smaller than the number of features we want per class:
                        multiplier = torch.ceil(torch.cuda.FloatTensor(
                            [max(1, sample_per_class / features_of_that_class.size(0))])).long().item()

                        features_of_that_class = features_of_that_class.repeat(multiplier, 1)

                        if i == 0:
                            features_to_return = features_of_that_class[:sample_per_class, :]
                            labels_to_return = s.repeat(sample_per_class)
                        else:
                            features_to_return = torch.cat(
                                (features_to_return, features_of_that_class[:sample_per_class, :]), dim=0)
                            labels_to_return = torch.cat((labels_to_return, s.repeat(sample_per_class)),
                                                         dim=0)

                    return features_to_return, labels_to_return
                else:
                    return torch.cuda.FloatTensor([]), torch.cuda.LongTensor([])


            img_seen_feat,   img_seen_label   = sample_train_data_on_sample_per_class_basis(
                train_seen_feat,train_seen_label,self.img_seen_samples )
            
            att_unseen_feat, att_unseen_label = sample_train_data_on_sample_per_class_basis(
                    novelclass_aux_data,
                    novel_corresponding_labels,self.att_unseen_samples )
            
              

            def convert_datapoints_to_z(features, encoder):
                if features.size(0) != 0:
                    mu_, logvar_ = encoder(features)
                    z = self.reparameterize(mu_, logvar_)
                    return z
                else:
                    return torch.cuda.FloatTensor([])
          
            z_seen_img   = convert_datapoints_to_z(img_seen_feat, self.encoder['resnet_features'])
            z_unseen_att = convert_datapoints_to_z(att_unseen_feat, self.encoder[self.auxiliary_data_source])
            #构成分类器的训练数据
            train_Z = [z_seen_img, z_unseen_att]
            train_L = [img_seen_label, att_unseen_label]

            train_X = [train_Z[i] for i in range(len(train_Z)) if train_Z[i].size(0) != 0]
            train_Y = [train_L[i] for i in range(len(train_L)) if train_Z[i].size(0) != 0]

            train_X = torch.cat(train_X, dim=0)
            train_Y = torch.cat(train_Y, dim=0)
            print("train_X shape:", train_X.shape)
            print("train_Y shape:", train_Y.shape)

        ############################################################
        ##### initializing the classifier and train one epoch
        ############################################################
       
        cls = classifier.CLASSIFIER(clf, train_X, train_Y, test_seen_X, test_seen_Y, test_novel_X,
                                    test_novel_Y,
                                    cls_seenclasses, cls_novelclasses,
                                    self.num_classes, self.device, self.lr_cls, 0.5, 1,
                                    self.classifier_batch_size,
                                    self.generalized)
        best_acc = 0.0
        best_k = 0
        for k in range(self.cls_train_epochs):
            if k > 0:
                if self.generalized:
                    cls.acc_seen, cls.acc_novel, cls.H = cls.fit()
                else:
                    cls.acc = cls.fit_zsl()

            if self.generalized:

                print('[%.1f]     novel=%.4f, seen=%.4f, h=%.4f , loss=%.4f' % (
                k, cls.acc_novel, cls.acc_seen, cls.H, cls.average_loss))

                
                if best_acc<cls.H.item():
                    best_acc = cls.H.item()
                    best_k = k

            else:
                #print('[%.1f]  acc=%.4f ' % (k, cls.acc))
                
                if best_acc<cls.acc.item():
                    best_acc = cls.acc.item()
                    best_k = k
        
        log_str = "curr_epoch:"+str(self.current_epoch)+"|curr_k:"+str(best_k)+"|best_acc:"+str(best_acc)+"\n"
        print(log_str)
        self.results_log_file.write(log_str)
        if self.best_result<best_acc:
            self.best_result = best_acc
            self.best_epoch = self.current_epoch
        
                
   
         

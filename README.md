# MCFA_pytorch

1. This code is for the paper of  Multi-level Cross-modal Feature Alignment via Contrastive Learning towards Zero-shot 
Classification of Remote Sensing Image Scenes, which is submitted to TGRS. This code is the implementation of the paper with the link: https://arxiv.org/abs/2306.06066.  
Please refer to the paper for better understanding our idea and the proposed method.

2. The code has been implemented based on that of CADA-VAE. The data can be available at link："https://pan.baidu.com/s/1-0lZaggJCeHJtOto8V35pw"
Extracted password："smhw". The data we have used for experiments are included, which can be seen in the folder of data.

3. We reproduced the visual features by using the ResNet18 which are also uploaded.

4. Moreover, we have found that there are significant differences in performance over different randomly seen/unseen data segments for these comparison methods. 
Thus, to ensure the fairness of the comparisons, we have first randomly selected five seen/unseen segments of the dataset. 
And all the following comparisons of each method are based on these selected segments.  All these segment files are also uploaded and can be used for further experiments.

5. All the parameters including the data path  are set in file of the single_experiment.py. The code should be run under the environment of Pytorch. When running the code, you can just run the single_experiment.py. 

6. If there are some problems, please email to liuchun through liuchun@henu.edu.cn. 


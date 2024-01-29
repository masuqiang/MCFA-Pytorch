# MCFA_pytorch

1. This code is for the paper of  Multi-level Cross-modal Feature Alignment via Contrastive Learning towards Zero-shot 
Classification of Remote Sensing Image Scenes, which is submitted to TGRS. 

2. The code has been implemented based on that of CADA-VAE. The data can be available at link "链接：https://pan.baidu.com/s/1-0lZaggJCeHJtOto8V35pw 
提取码：smhw". The data we have used for experiments are included, which can be seen in the folder of data.

3.This code is the implementation of the paper with the link: https://arxiv.org/abs/2306.06066.  
Please refer to the paper for better understanding our idea and the proposed method.

4.We have also not found  the visual features of the remote sensing image scenes from Internet, so we reproduced the visual features by using the ResNet18. 
These reproduced visual features are also uploaded.

5.Moreover, we have found that there are significant differences in performance over different randomly seen/unseen data segments for these comparison methods. 
Thus, to ensure the fairness of the comparisons, we have first randomly selected five seen/unseen segments of the dataset. 
And all the following comparisons of each method are based on these selected segments.  All these segment files are also uploaded and can be used for further experiments.

6. All the parameters including the data path  are set in file of the single_experiment.py. 

7. The code should be run under the environment of Pytorch. When running the code, you can just run the single_experiment.py. 

8. If there are some problems, please email to liuchun through liuchun@henu.edu.cn. 


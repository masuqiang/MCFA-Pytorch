import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.bias.data.fill_(0)

        nn.init.xavier_uniform_(m.weight,gain=0.5)


    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class encoder_template(nn.Module):

    def __init__(self,input_dim,latent_size,hidden_size_rule,device):
        super(encoder_template,self).__init__()



        if len(hidden_size_rule)==2:
            self.layer_sizes = [input_dim, hidden_size_rule[0], latent_size]
        elif len(hidden_size_rule)==3:
            self.layer_sizes = [input_dim, hidden_size_rule[0], hidden_size_rule[1] , latent_size]

        modules = []
        for i in range(len(self.layer_sizes)-2):

            modules.append(nn.Linear(self.layer_sizes[i],self.layer_sizes[i+1]))
            modules.append(nn.ReLU())

        self.feature_encoder = nn.Sequential(*modules)

        self._mu = nn.Linear(in_features=self.layer_sizes[-2], out_features=latent_size)


        self._logvar = nn.Linear(in_features=self.layer_sizes[-2], out_features=latent_size)


        self.apply(weights_init)

        self.to(device)


    def forward(self,x):

        h = self.feature_encoder(x)


        mu =  self._mu(h)
        logvar = self._logvar(h)

        return mu, logvar

class decoder_template(nn.Module):

    def __init__(self,input_dim,output_dim,hidden_size_rule,device):
        super(decoder_template,self).__init__()


        self.layer_sizes = [input_dim, hidden_size_rule[-1] , output_dim]

        self.feature_decoder = nn.Sequential(nn.Linear(input_dim,self.layer_sizes[1]),nn.ReLU(),nn.Linear(self.layer_sizes[1],output_dim))

        self.apply(weights_init)

        self.to(device)
    def forward(self,x):

        return self.feature_decoder(x)
        
        
        
        
      
class discriminator(nn.Module):
    def __init__(self, x_dim=2048, s_dim=300, layers='1200'):#1200 600
        super(discriminator, self).__init__()
        self.x_dim = x_dim
        self.s_dim = s_dim
        total_dim = x_dim + s_dim#总维度400
        layers = layers.split()#得到带字符串的列表['1200','600']
        fcn_layers = []

        for i in range(len(layers)):
            pre_hidden = int(layers[i-1])
            num_hidden = int(layers[i])
            if i == 0:
                fcn_layers.append(nn.Linear(total_dim, num_hidden))
                fcn_layers.append(nn.ReLU())
            else:
                fcn_layers.append(nn.Linear(pre_hidden, num_hidden))
                fcn_layers.append(nn.ReLU())#nn.ReLU作为一个层结构，必须添加到nn.Module容器中才能使用，而F.ReLU则作为一个函数调用。
#ReLU是将所有的负值都设为零，正值做线性变换并且范围不一定是从零到一
            if i == len(layers) - 1:
                fcn_layers.append(nn.Linear(num_hidden, 1))#输出一个分数
        # 顺序容器.模块将按照顺序存进sequential中，相当于一个包装起来的子模块集, 已经实现了forward（）方法，可以在forward中直接运行。
        # 而且里面的模块是按照顺序进行排列的，所以我们必须确保前一个模块的输出大小和下一个模块的输入大小是一致的。

        self.FCN = nn.Sequential(*fcn_layers)
        #nn.MSELoss均方损失函数
        self.mse_loss = nn.MSELoss()#引用模型自带的损失函数

    def forward(self, X, S):
        #torch.cat是将两个张量（tensor）拼接在一起,1表示横着拼接
        XS = torch.cat([X, S], 1)
        return self.FCN(XS)
        #forward（）实现了模型架构

    def dis_loss(self, X, Xp, S, Sp):
        true_scores = self.forward(X, S)
        fake_scores = self.forward(Xp, S)
        ctrl_socres = self.forward(X, Sp)
        
        return self.mse_loss(true_scores, 1) + self.mse_loss(fake_scores, 0) + self.mse_loss(ctrl_socres, 0)
#mse标准来测量输入x和目标y（0.1）中每个元素之间的均方误差（L2范数平方），并使其误差最小化



      
class discriminator_xs(nn.Module):
    def __init__(self, x_dim=2048,layers='1200'):#1200 600
        super(discriminator_xs, self).__init__()
        self.x_dim = x_dim
        
        layers = layers.split()#得到带字符串的列表['1200','600']
        fcn_layers = []

        for i in range(len(layers)):
            pre_hidden = int(layers[i-1])
            num_hidden = int(layers[i])
            if i == 0:
                fcn_layers.append(nn.Linear(self.x_dim, num_hidden))
                fcn_layers.append(nn.ReLU())
            else:
                fcn_layers.append(nn.Linear(pre_hidden, num_hidden))
                fcn_layers.append(nn.ReLU())#nn.ReLU作为一个层结构，必须添加到nn.Module容器中才能使用，而F.ReLU则作为一个函数调用。
#ReLU是将所有的负值都设为零，正值做线性变换并且范围不一定是从零到一
            if i == len(layers) - 1:
                fcn_layers.append(nn.Linear(num_hidden, 1))#输出一个分数
        # 顺序容器.模块将按照顺序存进sequential中，相当于一个包装起来的子模块集, 已经实现了forward（）方法，可以在forward中直接运行。
        # 而且里面的模块是按照顺序进行排列的，所以我们必须确保前一个模块的输出大小和下一个模块的输入大小是一致的。

        self.FCN = nn.Sequential(*fcn_layers)
        #nn.MSELoss均方损失函数
        self.mse_loss = nn.MSELoss()#引用模型自带的损失函数

    def forward(self, X):
        
        return self.FCN(X)
        #forward（）实现了模型架构

   


"""
######################################
##新增加的损失函数代码
######################################
def mcdd_loss(U):
    n, d = U.shape
    P = torch.eye(n)
    W = torch.ones(n, n)
    H = n*P-W
    H = H/n
    UT = U.transpose(0, 1)
    UT = torch.mm(UT, H)
    UT = torch.mm(UT, U)
    I = torch.eye(d)
    UT = UT - I
    UT = torch.norm(UT, dim=1)
    UT = torch.pow(UT, 2)
    return UT

"""
######################################
##新增加的损失函数代码
######################################
def cos_loss(zv, zs, l):
    import torch.nn.functional as F
    n, d = zv.shape

    zv = zv.repeat(n, 1, 1)  # x对行复制n份，列不变
    zv = zv.transpose(0, 1)
    zs = zs.repeat(n, 1, 1)
    zv = F.cosine_similarity(zv, zs, dim=2)#此时输出的zv的大小应该是n*n，每一行表示一个视觉特征与其它所有标签对应的语义特征的余弦相识度

    # 第二步：区分同类的和不同类的，根据标签来确定，因为不同视觉可能对应的类别标签是一样的, l为标签列表
    l = l.repeat(n, 1)
    l1 = l
    l = l.transpose(0, 1)
    l = l - l1#相减之后的矩阵每一行表示一个视觉样本的标签与其他样本标签的值相减，如果是同类，则对应元素为0，否则则不为0
    l[l != 0] = -1#不同类的余弦相识度应该是0,先做个标记
    l[l == 0] = 1#同类的余弦相识度应该是1
    l[l == -1] = 0#不同类的余弦相识度应该是0
    # 第三步
    zv = zv-l
    zv = torch.pow(zv, 2)
    #zv = zv.sum()/(n*n)
    zv = zv.sum()
    return zv



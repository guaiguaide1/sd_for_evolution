import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.init as init
from torch.autograd import Variable
import random 
import numpy as np 

class MLPDiffusion(nn.Module):    
    def __init__(self, d, n_steps):
        super(MLPDiffusion,self).__init__()
        num_units = d

        self.layer1 = nn.Linear(d, num_units)
        self.layer2 = nn.Linear(num_units, num_units)
        # self.layer3 = nn.Linear(num_units, num_units)
        self.layer4 = nn.Linear(num_units, d)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()                 # self.tanh = nn.Tanh()  relu更容易收敛
        # self.bn_layers = nn.ModuleList([nn.BatchNorm1d(num_units) for _ in range(3)])
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(num_units) for _ in range(2)])

        # self.step_embeddings = nn.ModuleList([nn.Embedding(n_steps,num_units) for _ in range(3)])
        self.step_embeddings = nn.ModuleList([nn.Embedding(n_steps,num_units) for _ in range(2)])

        # Initialize weights
        self._initialize_weights()

    def forward(self, x, t):
        for idx, (embedding_layer, bn_layer) in enumerate(zip(self.step_embeddings, self.bn_layers)):
            t_embedding = embedding_layer(t)
            # x = self.layer1(x) if idx == 0 else self.layer2(x) if idx == 1 else self.layer3(x)
            x = self.layer1(x) if idx == 0 else self.layer2(x)
            x += t_embedding
            x = bn_layer(x)
            x = self.relu(x)
        
        x = self.layer4(x)
        x = self.sigmoid(x)
        return x

    

class Diffusion(object):  # 注意：这里的batchsize和GAN里面的顺序不一样
    def __init__(self, dim, lr, epoches, batchsize=32):
        # 1. 实例化的参数
        self.dim = dim 
        self.batchsize = batchsize 
        self.lr = lr 
        self.epoches = epoches 


        # 2. 设置一些参数
        self.num_steps = 100    # 即T,对于步骤，一开始可以由beta, 分布的均值和标准差来共同确定
        self.betas = torch.linspace(-6, 6, self.num_steps)#制定每一步的beta, size:100
        self.betas = torch.sigmoid(self.betas)*(0.5e-2 - 1e-5)+1e-5   
        # beta是递增的，最小值为0.00001,最大值为0.005, sigmooid func
        # 像学习率一样的一个东西，而且是一个比较小的值，所以就有理由假设逆扩散过程也是一个高斯分布
        #计算alpha、alpha_prod、alpha_prod_previous、alpha_bar_sqrt等变量的值
        self.alphas = 1 - self.betas    # size: 100
        self.alphas_prod = torch.cumprod(self.alphas, 0)    # size: 100
        # 就是让每一个都错一下位
        self.alphas_prod_p = torch.cat([torch.tensor([1]).float(), self.alphas_prod[:-1]],0)  # p表示previous  
        # alphas_prod[:-1] 表示取出 从0开始到倒数第二个值
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)
        self.one_minus_alphas_bar_log = torch.log(1 - self.alphas_prod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod)

        assert self.alphas.shape == self.alphas_prod.shape == self.alphas_prod_p.shape ==\
        self.alphas_bar_sqrt.shape == self.one_minus_alphas_bar_log.shape\
        == self.one_minus_alphas_bar_sqrt.shape


        # 3.初始化去噪模型
        self.Denoise = MLPDiffusion(self.dim, self.num_steps)

        # 4.损失函数
        self.loss = nn.MSELoss()

        # 5.优化器
        # weight_decay=1e-5   添加L2正则化，权重衰减
        # self.optimizer = optim.Adam(self.Denoise.parameters(), lr=self.lr, weight_decay=1e-5)
        self.optimizer = optim.Adam(self.Denoise.parameters(), lr=self.lr)
    

    #前向加噪过程，计算任意时刻加噪后的xt，基于x_0和重参数化
    def q_x(self, x_0, t, center, cov):
        """可以基于x[0]得到任意时刻t的x[t]"""

        noise = np.random.multivariate_normal(center, cov, x_0.shape[0])  
        noise = torch.from_numpy(np.maximum(np.minimum(noise, np.ones(( x_0.shape[0], self.dim))),
                                             np.zeros(( x_0.shape[0], self.dim)))).float()

        # RuntimeWarning: covariance is not symmetric positive-semidefinite.
        # noise = torch.from_numpy(noise).float()

        # noise = torch.randn_like(x_0)   # noise是从某分布中生成的随机噪声
        alphas_t = self.alphas_bar_sqrt[t]
        alphas_1_m_t = self.one_minus_alphas_bar_sqrt[t]

        xt = alphas_t * x_0 + alphas_1_m_t * noise
        return xt #在x[0]的基础上添加噪声
        # 上面就可以通过x0和t来采样出xt的值

    def diffusion_loss_fn(self, x_0, center, cov):
        # n_steps为中的时间步数，这里是500步
        batch_size = x_0.shape[0]
        n_steps = self.num_steps

        x_0 = torch.from_numpy(x_0).float()

        t = torch.full((batch_size,), n_steps-1)
        t = t.unsqueeze(-1)

        xt = self.q_x(x_0, t, center, cov)

        output = self.Denoise(xt, t.squeeze(-1))  # 这里让模型直接预测x_0而不是噪声

        # 重构误差
        loss1 = (x_0 - output).square().mean()

        total_loss = loss1
        return total_loss
    
    def train(self, pop_dec, positive_samples):
        ''' 
        pop_dec: shape(32, 31)    用于训练的数据，这里只选前32个样本用于训练
        samples_pool.shape=(10, 31)是当前种群中表现最好的10个解，计算他们的均值和方差，用以生成随机噪声，即作为随机噪声的均值和方差
        '''
        self.Denoise.train()
        n, d = np.shape(pop_dec)
        indices = np.arange(n)  # indices=array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,..., 30, 31])
        
        center = np.mean(positive_samples, axis=0)  # (31,1)  axis=0，对第一个维度求均值    下面的 cov 矩阵提供了一个关于这10个样本在31个特征上相互关系的全面视图。
        cov = np.cov(positive_samples[:10, :].reshape((d, positive_samples[:10, :].size // d)))#  (10, 31)->(31, 10)  conv=(31,31)  np.cov 函数用于计算协方差矩阵   samples_pool.shape=(10, 31),   


        for epoch in range(self.epoches):
            loss = 0
            self.optimizer.zero_grad()
            loss = self.diffusion_loss_fn(pop_dec, center, cov)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.Denoise.parameters(), 1.)
            self.optimizer.step()
            # print("Epoch[{}], loss: {:.5f}".format(epoch, loss))

        random.shuffle(indices)
        pop_dec = pop_dec[indices, :]   # 感觉这里应该加上label = labels[indices, :]


    def p_sample_loop(self, x_T, center, cov):
        """从x[T]恢复x[T-1]、x[T-2]|...x[0]"""
        cur_x = x_T

        x_0 = self.p_sample(cur_x, self.num_steps - 1, center, cov)
        return x_0

    def p_sample(self, x, t, center, cov): # 参数重整化的过程
        """从x[t]采样t-1时刻的重构值，即从x[t]采样出x[t-1]"""
        t = torch.tensor([t])
        x_0 = self.Denoise(x,t)

        return x_0
    

    def generate(self, sample_noises, population_size):# population_size=100
        self.Denoise.eval()
        center = np.mean(sample_noises, axis=0).T
        cov = np.cov(sample_noises.T)

        noises = np.random.multivariate_normal(center, cov, population_size)
        noises = torch.from_numpy(np.maximum(np.minimum(noises, np.ones((population_size, self.dim))),
                                             np.zeros((population_size, self.dim)))).float()

        with torch.no_grad():
            decs= self.p_sample_loop(Variable(noises.cpu()).float(), center, cov).cpu().data.numpy()
    
        return decs 
    

import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.init as init
from torch.autograd import Variable
import random 
import numpy as np 
from sklearn.utils import resample
import torch.nn.functional as F
import math

class PositionalEmbedding(nn.Module):
    __doc__ = r"""Computes a positional embedding of timesteps.

    Input:
        x: tensor of shape (N)
    Output:
        tensor of shape (N, dim)
    Args:
        dim (int): embedding dimension
        scale (float): linear scale to be applied to timesteps. Default: 1.0
    """

    def __init__(self, dim, scale=1.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.scale = scale

    def forward(self, x):  # torch.size([128])
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = torch.outer(x * self.scale, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb



class MLPDiffusion(nn.Module):    
    def __init__(self, instance):
        # instance：数据集的id，根据数据集的不同来设置不同的hidden_channel
        super(MLPDiffusion, self).__init__()
        
        dim = [16, 56, 56, 80, 160]
        # dim = [24, 32, 32, 48, 64]
        input_channel = 1
        hidden_channel = dim[instance-1] 
        output_channel = 1
        base_channels = 16
        time_emb_scale = 1.0
        time_emb_dim = base_channels
        dropout_prob = 0.5


        # 1D卷积层
        self.conv1d = nn.Conv1d(in_channels=input_channel, out_channels=hidden_channel, kernel_size=5, padding=2)
        # 激活函数
        self.relu = nn.ReLU()
        # Dropout层
        self.dropout = nn.Dropout(p=dropout_prob)

        self.mid1 = nn.Conv1d(in_channels=hidden_channel, out_channels=hidden_channel, kernel_size=5, padding=2)
        self.mid2 = nn.Conv1d(in_channels=hidden_channel, out_channels=hidden_channel, kernel_size=5, padding=2)
        # 输出层
        self.output_layer = nn.Conv1d(in_channels=hidden_channel, out_channels=output_channel, kernel_size=5, padding=2)

        self.time_mlp = nn.Sequential(
            PositionalEmbedding(base_channels, time_emb_scale),
            # nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(base_channels, hidden_channel),
        )

    def forward(self, x, t):
        # 输入张量 x 的形状应为 (batch_size, input_channel, dim)
        # t的形状应该为(batch_size,)
        batch, _, len = x.shape
        # t_emb = self.time_mlp(t)  # (batch_size, outputchannel)

        # 1D卷积操作
        x = self.conv1d(x)
        x = self.relu(x)
        # Dropout
        x = self.dropout(x)
        # 中间层1
        x = self.mid1(x)
        x = self.relu(x)
        x = self.dropout(x)
        # 中间层2
        # x = self.mid2(x)
        # x = self.relu(x)
        # x = self.dropout(x)

        # 输出层
        output = self.output_layer(x) 

        output = output.view(batch, -1)
        output = F.softmax(output/2.0, dim=1)  # 使用softmax确保输出的每个维度的和为1
        
        return output


class Diffusion(object):  # 注意：这里的batchsize和GAN里面的顺序不一样
    def __init__(self, instance, dim, lr, epoches, batchsize=32):
        # 1. 实例化的参数
        self.dim = dim 
        self.batchsize = batchsize 
        self.lr = lr 
        self.epoches = epoches 

        # 2. 设置一些参数
        self.num_steps = 30    # 即T,对于步骤，一开始可以由beta, 分布的均值和标准差来共同确定
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
        self.Denoise = MLPDiffusion(instance)
    
        # 4.损失函数
        self.MSEloss = nn.MSELoss()

        # 5.优化器
        self.optimizer = optim.Adam(self.Denoise.parameters(), lr=self.lr, weight_decay=1e-5)
    

    #前向加噪过程，计算任意时刻加噪后的xt，基于x_0和重参数化
    def q_x(self, x_0, t, noise):
        """可以基于x[0]得到任意时刻t的x[t]"""

        alphas_t = self.alphas_bar_sqrt[t]
        alphas_1_m_t = self.one_minus_alphas_bar_sqrt[t]

        xt = alphas_t * x_0 + alphas_1_m_t * noise
        return xt #在x[0]的基础上添加噪声
        # 上面就可以通过x0和t来采样出xt的值

    def regularize_loss(self, l1_factor=0.005, l2_factor=0.005):
        l1_loss = 0
        l2_loss = 0
        for param in self.Denoise.parameters():
            l1_loss += torch.sum(torch.abs(param))
            l2_loss += torch.sum(param ** 2)
        
        return l1_factor * l1_loss + l2_factor * l2_loss



    # 计算指标辅助调用
    def ndset(self, A):
        dominated = torch.zeros(A.size(0), dtype=torch.bool)
        M = A[:, 0].unsqueeze(1)
        V = A[:, 1].unsqueeze(1)
        
        M_dom = (M >= M.T) & (V > V.T) | (M > M.T) & (V >= V.T)
        dominated = M_dom.sum(dim=1) > 0

        return A[~dominated]
    
    def spread(self, A):
        A = self.ndset(A)
        M = -A[:, 0]  # Since pytorch supports slicing, no need to loop through.
        V = A[:, 1]
        s = torch.sqrt((torch.max(M) - torch.min(M)) ** 2 + (torch.max(V) - torch.min(V)) ** 2)
        sp_value_clipped = torch.clamp(s, min=1e-8)
        sp_value_log = torch.log(sp_value_clipped * 1e3)
        # s = torch.log(1e3 * s + 1e-6)  # 添加1e-6是为了防止对0取对数
        return s

    # 反转世代距离
    def igd(self, A, M_P, V_P):
        A = self.ndset(A)
        M_A = -A[:, 0]
        V_A = A[:, 1]

        # 使用广播来计算所有的M_P和M_A之间，以及V_P和V_A之间的差异
        diff_M = M_P[:, None] - M_A[None, :]
        diff_V = V_P[:, None] - V_A[None, :]

        # 计算每个组合的距离
        distances = torch.sqrt(diff_M**2 + diff_V**2)

        # 对于每个M_P和V_P，找到最小的距离
        min_distances, _ = torch.min(distances, dim=1)

        # 计算平均值
        d_avg = torch.mean(min_distances)

        return d_avg


    # 初始化目标函数
    def objective(self, P, r, s, c):
        # 对于objective 1: -return
        M = -torch.matmul(P.double(), r)  # 结果形状为 [100]

        # 对于objective 2: risk
        temp = P * s  # Element-wise乘法
        V = torch.sum(temp.unsqueeze(2) * temp.unsqueeze(1) * c, dim=(1, 2))  # 先通过unsqueeze扩展张量的维度，然后进行element-wise乘法和求和操作

        # 将M和V组合为一个新的张量
        objs = torch.stack([M, V], dim=1)

        return objs


    def loss_function(self, x_0_p, output_p, r, s, c, mp, vp):

        # loss1: 正样本的重构损失 
        recon_loss =  nn.MSELoss()(x_0_p, output_p) 
        loss1 = recon_loss

        # objs = self.objective(output_p, r, s, c)
        # sp_value = self.spread(objs)
        # # epsilon = 1e-5
        # # scaled_sp = sp_value * 1e6 + epsilon # spread越大越好
        # loss2 =  -sp_value    # 因为spread是越小越好，所以要取负号
        # print("loss2:{:.9f}".format(sp_value.item()))
        
        # 正则化（可选）
        reg_loss = self.regularize_loss()  # 您可以选择L1、L2或其他形式的正则化
        
        total_loss = loss1  + reg_loss
        return total_loss
 
    def diffusion_loss_fn(self, x_0_p, r, s, c, mp, vp, noise):
        # x_0_p: positive_samples    (batch, dim) = (32, 31)
        # 对正样本处理，n_steps为中的时间步数，这里是30步
        batch_size_p = x_0_p.shape[0]

        t_p = torch.randint(0, self.num_steps, size=(batch_size_p // 2,))
        t_p = torch.cat([t_p, self.num_steps - 1 - t_p], dim=0)  # t_p.shape=torch.Size([32])
        t = t_p.unsqueeze(-1)# t_p.shape=torch.Size([32, 1])

        # 前向加噪过程
        xt_p = self.q_x(x_0_p, t, noise)  # (batch,dim) = (32,31)

        noise_x = xt_p.unsqueeze(1)  # (batch, channel, dim) = (32, 1, 31)
        # model预测噪声  noise_x.shape=(b, c, d)=(32, 1, 31)   t_p.shape=(b,)=(32,)
        output_p= self.Denoise(noise_x, t_p)  # 这里让模型直接预测x_0而不是噪声, output.shape=(batch, dim)
        total_loss = self.loss_function(x_0_p, output_p, r, s, c, mp, vp)
        return total_loss
    
    def train(self, population, r, s, c, mp, vp):
        self.Denoise.train()
        n, d = np.shape(population)  # n=100, d=31

        population_size = n

        # 获得种群中所有解的均值和协方差矩阵
        center = np.mean(population, axis=0).T
        cov = np.cov(population.T)
        # 以均值何协方差矩来生成噪声
        noises = np.random.multivariate_normal(center, cov, population_size)
        noises = torch.from_numpy(np.maximum(np.minimum(noises, np.ones((population_size, self.dim))),
                                             np.zeros((population_size, self.dim)))).float()  # (100,31)


        
        indices = np.arange(n)  # indices=array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,..., 30, 31])
        iter_no = (n + self.batchsize - 1) // self.batchsize
        x0 = torch.from_numpy(population).float()   # 转为tensor   (100, 31)

        # 将 numpy 数组转换为张量并使用squeeze方法移除大小为1的维度
        r = torch.from_numpy(r).squeeze()   # 将r从 numpy (31,1)转为torch.size([31])
        s = torch.tensor(s).squeeze()
        c = torch.tensor(c)                 # 将c从 numpy (31,31)转为torch.size([31, 31])
        mp = torch.tensor(mp)               # 将mp从list转为 torch.size([2000])
        vp = torch.tensor(vp)               # 将vp从list转为 torch.size([2000])


        for epoch in range(self.epoches):
            losses = 0
            for iteration in range(iter_no):
                self.optimizer.zero_grad()
                given_p = x0[iteration * self.batchsize: (1 + iteration) * self.batchsize, :]
                given_noise = noises[iteration * self.batchsize: (1 + iteration) * self.batchsize, :]
                loss = self.diffusion_loss_fn(given_p, r, s, c, mp, vp, given_noise)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.Denoise.parameters(), 5)
                self.optimizer.step()
                losses += loss
            random.shuffle(indices)
            x0 = x0[indices, :]   # 感觉这里应该加上label = labels[indices, :]
      
    def p_sample_loop(self, x_T):
        """从x[T]恢复x[T-1]、x[T-2]|...x[0]"""
        x_0 = self.p_sample(x_T)
        return x_0

    def p_sample(self, x): # 参数重整化的过程
        """从x[t]采样t-1时刻的重构值，即从x[t]采样出x[t-1]"""
        t = torch.full([x.shape[0]], self.num_steps-1)  #  torch.full([shape], value)   (batch,)
        noise = x.unsqueeze(1)  # (batch, channel, dim) = (32, 1, 31)
        x_0 = self.Denoise(noise, t)
        return x_0

    def generate(self, population, population_size):# population_size=100
        self.Denoise.eval()

        # 获得种群中所有解的均值和协方差矩阵
        center = np.mean(population, axis=0).T
        cov = np.cov(population.T)
        # 以均值何协方差矩来生成噪声
        noises = np.random.multivariate_normal(center, cov, population_size)
        noises = torch.from_numpy(np.maximum(np.minimum(noises, np.ones((population_size, self.dim))),
                                             np.zeros((population_size, self.dim)))).float()

        with torch.no_grad():
            decs = self.p_sample_loop(Variable(noises).float()).cpu().data.numpy()
        return decs 
    

 

    

# DiffGan
import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.init as init
from torch.autograd import Variable
import random 
import numpy as np 
import torch.nn.functional as F
from sklearn.utils import resample


class DiffGenerator(nn.Module):    
    def __init__(self, d, n_steps):
        super(DiffGenerator, self).__init__()
        num_units = d

        self.layer1 = nn.Linear(d, num_units, bias=True)
        self.layer2 = nn.Linear(num_units, num_units, bias=True)
        self.layer3 = nn.Linear(num_units, d, bias=True)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(num_units) for _ in range(2)])
        self.step_embeddings = nn.ModuleList([nn.Embedding(n_steps, num_units) for _ in range(2)])

        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m == self.layer3:
                    # Xavier initialization for the layer with sigmoid activation
                    init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        init.constant_(m.bias, 0)
                else:
                    # He initialization for layers with ReLU activation
                    init.kaiming_uniform_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x, t):
        for idx, (embedding_layer, bn_layer) in enumerate(zip(self.step_embeddings, self.bn_layers)):
            t_embedding = embedding_layer(t)
            x = self.layer1(x) if idx == 0 else self.layer2(x)
            x += t_embedding
            x = self.relu(x)
            x = bn_layer(x)  # 一般来说bn在relu之后
            x = self.dropout(x)
        
        x = self.layer3(x)
        x = F.softmax(x, dim=1)  # 使用softmax确保输出的每个维度的和为1
        # x = self.sigmoid(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, d):  # d=31
        super(Discriminator, self).__init__()
        self.linear1 = nn.Linear(d, d, bias=True)
        self.linear2 = nn.Linear(d, 1, bias=True)

    def forward(self, dec):   # （8， 31）
        x = torch.tanh(self.linear1(dec)) # (8, 31)->(8, 31)
        x = torch.sigmoid(self.linear2(x)) # (8, 31)->(8, 1)
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
        self.G = DiffGenerator(self.dim, self.num_steps)
        self.D = Discriminator(self.dim)

        # 4.损失函数
        # self.loss = nn.MSELoss()
        self.BCE_loss = nn.BCELoss()

        # 5.优化器
        # weight_decay=1e-5   添加L2正则化，权重衰减     self.lr = 0.005
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=self.lr, weight_decay=1e-5)
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=self.lr, weight_decay=1e-5)

    #前向加噪过程，计算任意时刻加噪后的xt，基于x_0和重参数化
    def q_x(self, x_0, t, center, cov):
        """可以基于x[0]得到任意时刻t的x[t]"""

        # noise是从某分布中生成的随机噪声
        noise = np.random.multivariate_normal(center, cov, x_0.shape[0])  
        noise = torch.from_numpy(np.maximum(np.minimum(noise, np.ones(( x_0.shape[0], self.dim))),
                                             np.zeros(( x_0.shape[0], self.dim)))).float()


        alphas_t = self.alphas_bar_sqrt[t]
        alphas_1_m_t = self.one_minus_alphas_bar_sqrt[t]

        xt = alphas_t * x_0 + alphas_1_m_t * noise
        return xt #在x[0]的基础上添加噪声
        # 上面就可以通过x0和t来采样出xt的值

    def train(self, pop_dec, labels, positive_samples, negative_samples):
        # def train(self, pop_dec, samples_pool):
        ''' 
        pop_dec: shape(100, 31)   
        positive_samples.shape=(10, 31)是当前种群中表现最好的10个解，计算他们的均值和方差，用以生成随机噪声，即作为随机噪声的均值和方差
        
        labels: 标注哪些是正样本哪些是负样本
        '''
        self.G.train()
        self.D.train()
        n, d = np.shape(pop_dec)
        indices = np.arange(n)  # indices=array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,..., 30, 31])
        
        center = np.mean(positive_samples[:10, :], axis=0)  # (31,1)  axis=0，对第一个维度求均值    下面的 cov 矩阵提供了一个关于这10个样本在31个特征上相互关系的全面视图。
        cov = np.cov(positive_samples[:10, :].reshape((d, positive_samples[:10, :].size // d)))#  (10, 31)->(31, 10)  conv=(31,31)  np.cov 函数用于计算协方差矩阵   samples_pool.shape=(10, 31),   

        k = 10  # 一般训练判别器的轮数多余训练生成器的轮数
        negative_samples = torch.from_numpy(negative_samples).float()

        for epoch in range(self.epoches):
            g_train_losses = 0

            # 训练判别器
            
            # 先训练判别器
            self.D.zero_grad()

            given_x = pop_dec 
            given_y = labels 
            given_x_ = Variable(torch.from_numpy(given_x)).float()
            given_y = Variable(torch.from_numpy(given_y)).float()

            d_results_real = self.D(given_x_)

            # 定义目标样本数量
            target_samples = 50

            # # 过采样正样本至50个
            x0 = resample(positive_samples, 
                            replace=True, 
                            n_samples=target_samples,
                            random_state=123)

            x0 = torch.from_numpy(x0).float()
            batch_size = x0.shape[0]     # batch_size = 50
            n_steps = self.num_steps

            t = torch.full((batch_size,), n_steps-1)
            t = t.unsqueeze(-1)
            # 对x0进行加噪
            xt = self.q_x(x0, t, center, cov)  # 得到加噪后的xt
            fake_y = Variable(torch.zeros((batch_size, 1)))  # 用于训练判别器的正样本的标签
                
            g_results = self.G(xt, t.squeeze(-1))  # 用generater对xt进行去噪

            d_results_fake = self.D(g_results.detach())
            d_train_loss = self.BCE_loss(d_results_real, given_y) + \
                            self.BCE_loss(d_results_fake, fake_y)
                
            d_train_loss.backward()
            self.D_optimizer.step()


            # 训练生成器
            self.G.zero_grad()
            fake_y = Variable(torch.ones((batch_size, 1)))  # 用于训练生成器的正样本的标签
            d_results = self.D(g_results)

            # Loss1: 正样本的重构误差：
            Loss1 = (g_results - x0).square().mean()

            # Loss2的第一种选择: 与负样本的L2距离
            # 首先计算了每个正样本与所有负样本之间的欧氏距离，然后选择了每个正样本的最差匹配
            distances = torch.norm(g_results.unsqueeze(1) - negative_samples, dim=2)
            # 选择每个正样本的最差匹配（距离最大的负样本）
            worst_matches = torch.max(distances, dim=1)[0]
            # 定义一个损失函数来最小化最差匹配的距离
            Loss2 = torch.mean(worst_matches)

            # Loss2的第二种选择：计算了每个正样本与所有负样本之间的欧氏距离
            # differences = g_results.unsqueeze(1) - negative_samples
            # Loss2 = -torch.mean((differences**2).sum(dim=2))

            # Loss3: 计算重构正样本之间的距离
            distances = torch.cdist(g_results, g_results)
            # 将对角线上的值（每个解与自身的距离）设置为0 
            mask = torch.eye(len(g_results)) == 1
            distances.masked_fill(mask, 0)
            # 计算每个解的多样性度量，即与其他解的平均距离
            diversity_measures = distances.sum(1) / (len(g_results) - 1)
            # 计算整体的多样性度量
            overall_diversity = -diversity_measures.mean()
            alpha = 5

            # Loss4:BCELoss
            loss4 = self.BCE_loss(d_results, fake_y)

            # total loss
            g_train_loss = Loss1 + Loss2 + alpha * overall_diversity + loss4
            
            g_train_loss.backward()
            self.G_optimizer.step()

            g_train_losses = g_train_loss

            # print("Epoch[{}], loss: {:.5f}".format(epoch, g_train_losses))


            random.shuffle(indices)
            pop_dec = pop_dec[indices, :]   # 感觉这里应该加上label = labels[indices, :]
            labels = labels[indices, :] 

    def p_sample_loop(self, x_T):
        # """从x[T]恢复x[T-1]、x[T-2]|...x[0]"""
        cur_x = x_T

        x_0 = self.p_sample(cur_x, self.num_steps - 1)
        return x_0

    def p_sample(self, x, t): # 参数重整化的过程
        """从x[t]采样t-1时刻的重构值，即从x[t]采样出x[t-1]"""
        t = torch.tensor([t])
        x_0 = self.G(x,t)
        return x_0 
    

    def generate(self, sample_noises, population_size):# population_size=100
        self.G.eval()
        center = np.mean(sample_noises, axis=0).T
        cov = np.cov(sample_noises.T)

        noises = np.random.multivariate_normal(center, cov, population_size)
        noises = torch.from_numpy(np.maximum(np.minimum(noises, np.ones((population_size, self.dim))),
                                             np.zeros((population_size, self.dim)))).float()

        with torch.no_grad():
            decs= self.p_sample_loop(Variable(noises.cpu()).float()).cpu().data.numpy()
    
        return decs 
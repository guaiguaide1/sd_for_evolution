import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.init as init
from torch.autograd import Variable
import random 
import numpy as np 
from sklearn.utils import resample
import torch.nn.functional as F
import os 

# 矩阵X求梯度，根据梯度来进行数据增强

def gradient_M(r):  
    """计算解x相对于目标M的梯度"""
    return -r.T

def gradient_V(X, s, c):
    """计算解x相对于目标V的梯度"""
    s_expanded = s.T  # 使s的形状为(1, 31)
    inter_result = s_expanded * s_expanded * X  # 逐元素乘法
    return 2 * np.tensordot(inter_result, c, axes=([1], [0]))  # 矩阵乘法

def compute_gradients(X, r, s, c):
    grad_M = gradient_M(r)
    grad_V = gradient_V(X, s, c)
    return grad_M, grad_V
def perturb_solution_along_gradient(X, r, s, c, alpha=0.2):
    """沿着梯度方向进行小幅度扰动"""
    grad_M_val, grad_V_val = compute_gradients(X, r, s, c)
    
    # 根据两个梯度更新解x。这里的alpha是一个学习率参数，用于控制扰动的大小
    new_X = X - alpha * (grad_M_val + 3 * grad_V_val)   # [M, V] = [return, risk]
    
    # 确保解的每一维数值在[0, 1]之间
    new_X = np.clip(new_X, 0, 1)
    
    # 确保解的31维之和为1
    new_X /= new_X.sum(axis=1, keepdims=True)
    
    return new_X

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
        self.dropout = nn.Dropout(p=0.5)  # p 是 dropout 的概率，通常设置为0.5


        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # He initialization for layers with ReLU activation
                init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
                # if m == self.layer4:
                #     # Xavier initialization for the layer with sigmoid activation
                #     init.xavier_uniform_(m.weight)
                #     if m.bias is not None:
                #         init.constant_(m.bias, 0)
                # else:
                #     # He initialization for layers with ReLU activation
                #     init.kaiming_uniform_(m.weight, nonlinearity='relu')
                #     if m.bias is not None:
                #         init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x, t):
        for idx, (embedding_layer, bn_layer) in enumerate(zip(self.step_embeddings, self.bn_layers)):
            t_embedding = embedding_layer(t)
            # x = self.layer1(x) if idx == 0 else self.layer2(x) if idx == 1 else self.layer3(x)
            x = self.layer1(x) if idx == 0 else self.layer2(x)
            x += t_embedding  
            x = self.dropout(x)  # 添加 Dropout
            x = bn_layer(x)
            x = self.relu(x)
        
        x = self.layer4(x)
        x = F.softmax(x/1.5, dim=1)  # 使用softmax确保输出的每个维度的和为1
        # x = self.sigmoid(x)
        return x

# 双塔网络结构（Siamese Networks）是指两个完全相同的子网络并行运行，共享相同的权重，
# 并对两个输入产生两个输出。这种网络的目的是比较这两个输出，通常用于计算两个输入之间的相似性或差异。
# 这种网络结构常用于一系列任务，如人脸验证、签名验证和图像相似性匹配。
class SiameseDiffModel(nn.Module):    
    def __init__(self, d, n_steps):
        super(SiameseDiffModel, self).__init__()
        self.diffModel = MLPDiffusion(d, n_steps)

        # 初始化可学习的系数 [0.01, 5]     最开始的初始化1, 1, 0.05
        self.alpha = nn.Parameter(torch.tensor(20.))  # 如果是6的话得写成小数形式：6.  
        self.beta = nn.Parameter(torch.tensor(5.))  # 如果是6的话得写成小数形式：6. 
        self.gamma = nn.Parameter(torch.tensor(5.)) 

    def forward_one(self, x, t):
        return self.diffModel(x, t)

    def forward(self, x1, t1, x2, t2): 
        # x1和x2可以是正样本和负样本
        output1 = self.forward_one(x1, t1)
        output2 = self.forward_one(x2, t2)
        return output1, output2


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
        # self.Denoise = MLPDiffusion(self.dim, self.num_steps)
        self.Denoise = SiameseDiffModel(self.dim, self.num_steps)
        # 检查 'best_model.pth' 是否存在
        # if os.path.exists('best_model.pth'):
        #     # 如果文件存在，加载模型参数
        #     checkpoint = torch.load('best_model.pth')
        #     self.Denoise.load_state_dict(checkpoint['state_dict'])

        # 4.损失函数
        self.MSEloss = nn.MSELoss()

        # 5.优化器
        # weight_decay=1e-5   添加L2正则化，权重衰减
        # self.optimizer = optim.Adam(self.Denoise.parameters(), lr=self.lr, weight_decay=1e-5)
        self.optimizer = optim.Adam(self.Denoise.parameters(), lr=self.lr, weight_decay=1e-5)
    

    #前向加噪过程，计算任意时刻加噪后的xt，基于x_0和重参数化
    def q_x(self, x_0, t, center, cov):
        """可以基于x[0]得到任意时刻t的x[t]"""

        noise = np.random.multivariate_normal(center, cov, x_0.shape[0]) 
        noise = self.softmax(noise)
        noise =  torch.from_numpy(noise).float()
        # noise = torch.from_numpy(np.maximum(np.minimum(noise, np.ones(( x_0.shape[0], self.dim))),
        #                                      np.zeros(( x_0.shape[0], self.dim)))).float()

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
    
    def population_entropy(self, population):
        """
        计算种群熵来衡量种群的多样性。
        :param population: 一个形状为 [N, D] 的张量，其中 N 是种群大小，D 是解的维度。
        :return: 一个标量张量表示种群的熵。
        """
        N, D = population.shape
        
        # 计算每个个体与种群中其他所有个体的欧氏距离的平方和
        distances = torch.norm(population[:, None] - population, dim=2, p=2) ** 2
        
        # 使用高斯函数进行归一化
        p = torch.exp(-distances / (2 * distances.var(dim=1, keepdim=True)))
        # p /= p.sum(dim=1, keepdim=True)
        p = p / p.sum(dim=1, keepdim=True)

        
        # 计算整个种群的熵
        entropy = -torch.sum(p * torch.log2(p + 1e-10)) / N  # 加上一个小常数防止对零取对数

        return entropy
    
    

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


    def loss_function(self, x_0_p, x_0_n, output_p, output_n, r, s, c, mp, vp,  margin=0.25):# margin=0.5
        '''                        
        x_0_p: positive_samples                  # 上一个margin为0.8效果不好
        x_0_n: negative_samples
        output_p:
        output_n:
        '''

        # 使用ReLU确保lambda_weight始终为正
        # 为了确保正样本的损失（loss_positive）在整体损失中占有更大的权重
        alpha = torch.clamp(torch.relu(self.Denoise.alpha), min=1, max=30)
        beta = torch.clamp(torch.relu(self.Denoise.beta), min=0.01, max=20)
        gamma = torch.clamp(torch.relu(self.Denoise.gamma), min=0.01, max=20)

        # loss1: 正样本的重构损失 
        recon_loss =  nn.MSELoss()(x_0_p, output_p) - 1.025 * nn.MSELoss()(x_0_n, output_n)   
        loss1 = recon_loss
        
        # loss2: 对比损失  
        positive_dist = torch.norm(output_p - x_0_p, dim=1)   # 正样本的L1损失
        negative_dist = torch.norm(output_p - x_0_n, dim=1)
        
        # distances = torch.cdist(output_p, output_n)
        # average_distances = distances.mean(dim=1)  # 计算每个x_output与所有x_negative之间的平均距离
        # overall_average_distance = average_distances.mean()  # 你可能还想计算所有的平均距离的总平均，以用于损失
        # negative_dist = overall_average_distance # 这个值可以被用作或者与其他损失结合作为对比损失
        contrast_loss = torch.clamp(margin + positive_dist - negative_dist, min=0).mean()
        loss2 = contrast_loss
        # contrast_loss = torch.clamp(margin + positive_dist - negative_dist, min=0).mean()
        # contrast_loss = -negative_dist
        
        # loss3: 计算正样本两两之间的距离
        # distances = torch.cdist(output_p, output_p)
        # mask = torch.eye(len(output_p)) == 1# 将对角线上的值(每个解与自身的距离)设置为0
        # distances.masked_fill(mask, 0)
        # diversity_measures = distances.sum(1) / (len(output_p) - 1)#计算每个解的多样性度量，即与其他解的平均距离
        # overall_diversity = -diversity_measures.mean()# 计算整体的多样性度量
        # loss3 = overall_diversity  

        # loss3 = - self.population_entropy(output_p)

        # output_p 和 x_0_n的距离也要尽可能地远
        # differences = output_p.unsqueeze(1) - x_0_n
        # loss_negative = -torch.mean((differences ** 2).sum(dim=2))

        # loss4
        # objs = self.objective(output_p, r, s, c)  # 把求得的解带入得到return, risk
        # sp_value = self.spread(objs)
        # # log_igd = -torch.log(indicator_value + 1e-8)  # 加上一个小常数以防止对数为负无穷
        # # loss4 = log_igd

        # epsilon = 1e-6
        # scaled_sp = sp_value * 1e2 + epsilon # spread越大越好
        # loss4 = - scaled_sp 
        # print("spread_value {:.9f}".format(scaled_sp.item()))

        # loss4 = indicator_value
        # print("indicator_value", loss4)
        # print("indicator_value {:.9f}".format(loss4.item()))

        
        # 正则化（可选）
        reg_loss = self.regularize_loss()  # 您可以选择L1、L2或其他形式的正则化
        
        # total_loss = alpha * recon_loss + beta * contrast_loss + gamma * (loss3) + 0.275 * loss_negative + 0.5 * reg_loss
        total_loss = 2 * loss1 + 0.5 * reg_loss
        print("total_loss {:.9f}".format(total_loss))
        return total_loss
 

    def diffusion_loss_fn(self, x_0_p, x_0_n, center, cov, r, s, c, mp, vp):
        # x_0_p: positive_samples
        # x_0_n: negative_samples

        # 对正样本处理，n_steps为中的时间步数，这里是100步
        batch_size_p = x_0_p.shape[0]
        n_steps = self.num_steps
        x_0_p = torch.from_numpy(x_0_p).float()

        t_p = torch.randint(0, n_steps, size=(batch_size_p // 2,))
        t_p = torch.cat([t_p, n_steps - 1 - t_p], dim=0)
        # t_p = torch.full((batch_size_p,), n_steps-1)
        t_p = t_p.unsqueeze(-1)
        xt_p = self.q_x(x_0_p, t_p, center, cov)

        # 对负样本处理
        
        batch_size_n = x_0_n.shape[0]
        n_steps = self.num_steps
        x_0_n = torch.from_numpy(x_0_n).float()

        t_n = torch.randint(0, n_steps, size=(batch_size_n // 2,))
        t_n = torch.cat([t_n, n_steps - 1 - t_n], dim=0)
        # t_n = torch.full((batch_size_n,), n_steps-1)
        t_n = t_n.unsqueeze(-1)
        xt_n = self.q_x(x_0_n, t_n, center, cov)


        output_p, output_n = self.Denoise(xt_p, t_p.squeeze(-1),  xt_n, t_n.squeeze(-1))  # 这里让模型直接预测x_0而不是噪声

        total_loss = self.loss_function(x_0_p, x_0_n, output_p, output_n, r, s, c, mp, vp)
    
        return total_loss
    
    def train(self, positive_samples, negative_samples, r, s, c, mp, vp):
        ''' 
        pop_dec: shape(32, 31)    用于训练的数据，这里只选前32个样本用于训练
        samples_pool.shape=(10, 31)是当前种群中表现最好的10个解，计算他们的均值和方差，用以生成随机噪声，即作为随机噪声的均值和方差
        '''
        self.Denoise.train()
        # n, d = np.shape(positive_samples)
        # indices = np.arange(n)  # indices=array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,..., 30, 31])
        
        center = np.mean(positive_samples, axis=0)  # (31,1)  axis=0，对第一个维度求均值    下面的 cov 矩阵提供了一个关于这10个样本在31个特征上相互关系的全面视图。
        cov = np.cov(positive_samples[:10, :].reshape((self.dim, positive_samples[:10, :].size // self.dim)))#  (10, 31)->(31, 10)  conv=(31,31)  np.cov 函数用于计算协方差矩阵   samples_pool.shape=(10, 31),   

        # negative_samples = torch.from_numpy(negative_samples).float()

        # 创建一个空的 numpy 数组, 数据增强操作
        combined_perturb_x = np.array([])
        alphas = [0.1 * i for i in range(1, 21, 2)]   # 10个
        for i in alphas:
            perturb_x = perturb_solution_along_gradient(positive_samples, r, s, c, i)
            
            # 将 perturb_x 堆叠到 combined_perturb_x 中
            if combined_perturb_x.size == 0:
                combined_perturb_x = perturb_x
            else:
                combined_perturb_x = np.vstack((combined_perturb_x, perturb_x))

        # 上采样负样本至100个
        upsample_negative_samples = resample(negative_samples, 
                                        replace=True, 
                                        n_samples=100,
                                        random_state=123)
        
        n, d = np.shape(combined_perturb_x)
        indices = np.arange(n)  # indices=array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,..., 30, 31])
        iter_no = (n + self.batchsize - 1) // self.batchsize

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
                given_p = combined_perturb_x[iteration * self.batchsize: (1 + iteration) * self.batchsize, :]
                given_n = upsample_negative_samples[iteration * self.batchsize: (1 + iteration) * self.batchsize, :]
                # loss = self.diffusion_loss_fn(positive_samples, negative_samples, center, cov)
                # loss = self.diffusion_loss_fn(X_resampled, downsampled_negative_samples, center, cov)
                loss = self.diffusion_loss_fn(given_p, given_n, center, cov, r, s, c, mp, vp)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.Denoise.parameters(), 0.5)
                self.optimizer.step()
                losses += loss
                # print("Epoch[{}], loss: {:.5f}".format(epoch, loss))

            # if epoch % 5 == 0:
            #     torch.save({
            #         'state_dict': self.Denoise.state_dict(),
            #     }, 'best_model.pth')


            random.shuffle(indices)
            combined_perturb_x = combined_perturb_x[indices, :]   # 感觉这里应该加上label = labels[indices, :]
            upsample_negative_samples = upsample_negative_samples[indices, :]
        # torch.save({
        #             'state_dict': self.Denoise.state_dict(),
        #         }, 'best_model.pth')

    def p_sample_loop(self, x_T, center, cov):
        """从x[T]恢复x[T-1]、x[T-2]|...x[0]"""
        cur_x = x_T

        x_0 = self.p_sample(cur_x, self.num_steps - 1, center, cov)
        return x_0

    def p_sample(self, x, t, center, cov): # 参数重整化的过程
        """从x[t]采样t-1时刻的重构值，即从x[t]采样出x[t-1]"""
        t = torch.tensor([t])
        x_0 = self.Denoise.diffModel(x,t)

        return x_0
    

    def generate(self, sample_noises, population_size):# population_size=100
        self.Denoise.eval()
        center = np.mean(sample_noises, axis=0).T  # (31,)
        cov = np.cov(sample_noises.T)  # (31,31)


        # population_size = int(population_size / 2)

        noises = np.random.multivariate_normal(center, cov, population_size)   #(100,31)
        noises = self.softmax(noises)
        noises = torch.from_numpy(noises).float()
        # noises = torch.from_numpy(np.maximum(np.minimum(noises, np.ones((population_size, self.dim))),
                                            #  np.zeros((population_size, self.dim)))).float()

        with torch.no_grad():
            part1 = self.p_sample_loop(Variable(noises.cpu()).float(), center, cov).cpu()

            # decs= self.p_sample_loop(Variable(noises.cpu()).float(), center, cov).cpu().data.numpy()
            combined = torch.cat((part1, noises), dim=0)
            decs = combined
    
        return decs 
    

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True)) # for numerical stability
        return e_x / e_x.sum(axis=1, keepdims=True)



    

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import random
import numpy as np

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

class Generator(nn.Module):
    
    def __init__(self, d, n_noise):  # 1-d vector   d=31, n_noise=31
        super(Generator, self).__init__()
        self.linear1 = nn.Linear(n_noise, d, bias=True)
        self.bn1 = nn.BatchNorm1d(d)
        self.linear2 = nn.Linear(d, d, bias=True)
        self.bn2 = nn.BatchNorm1d(d)
        self.linear3 = nn.Linear(d, d, bias=True)
        self.bn3 = nn.BatchNorm1d(d)

    
    def forward(self, noise):  # noise=(8, 31)
        x = torch.tanh(self.bn1(self.linear1(noise)))  # (8,31)->(8,31)
        x = torch.tanh(self.bn2(self.linear2(x)))      # (8,31)->(8,31)
        x = torch.sigmoid(self.bn3(self.linear3(x)))   # (8,31)->(8,31)
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


class GAN(object):# d=31, batchsize=8, lr=0.0001, epoches=200, n_noise=31
    def __init__(self, d, batchsize, lr, epoches, n_noise):   
        self.d = d
        self.n_noise = n_noise
        self.BCE_loss = nn.BCELoss()
        self.G = Generator(self.d, self.n_noise)
        self.D = Discriminator(self.d)
        # self.G = Generator(self.d, self.n_noise).to(device)
        # self.D = Discriminator(self.d).to(device)
        self.G.cpu()
        self.D.cpu()
        self.G_optimizer = optim.Adam(self.G.parameters(), 4*lr)
        self.D_optimizer = optim.Adam(self.D.parameters(), lr)
        self.epoches = epoches
        self.batchsize = batchsize

    def train(self, pop_dec, labels, samples_pool, r, s, c):  # pop_dec.shape=(100, 31), labels.shape=(100, 1), samples_pool.shape=(10, 31)
        self.D.train()     # samples_pool，是当前种群中表现最好的10个解，计算他们的均值和方差，用以生成随机噪声，即作为随机噪声的均值和方差
        self.G.train()
        
        center = np.mean(samples_pool, axis=0)  # (31,1)  axis=0，对第一个维度求均值    下面的 cov 矩阵提供了一个关于这10个样本在31个特征上相互关系的全面视图。
        cov = np.cov(samples_pool[:10, :].reshape((self.d, samples_pool[:10, :].size // self.d)))#  (10, 31)->(31, 10)  conv=(31,31)  np.cov 函数用于计算协方差矩阵   samples_pool.shape=(10, 31),   

        # 创建一个空的 numpy 数组, 数据增强操作,生成200个新的正样本
        combined_perturb_x = np.array([])
        alphas = [0.1 * i for i in range(1, 21)]
        for i in alphas:
            perturb_x = perturb_solution_along_gradient(samples_pool, r, s, c, i)
            
            # 将 perturb_x 堆叠到 combined_perturb_x 中
            if combined_perturb_x.size == 0:
                combined_perturb_x = perturb_x
            else:
                combined_perturb_x = np.vstack((combined_perturb_x, perturb_x))

                n, d = np.shape(combined_perturb_x)
                
        labels_perturb_x = np.ones((200,1))
        labels_aug = np.vstack((labels, labels_perturb_x))

        pop_dec_aug = np.vstack((pop_dec, combined_perturb_x))
        n, d = np.shape(pop_dec_aug)  # n=300,  d=31
        indices = np.arange(n)  # indices=array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,..., 98, 99])
        iter_no = (n + self.batchsize - 1) // self.batchsize  # batchsize=8   n=100  iter_no:代表着在给定的设置中，需要多少批（batch）迭代来处理所有 n 个样本。其中，每个批的大小由 self.batchsize（在这个例子中是8）确定。


        for epoch in range(self.epoches): # epoches=200
            g_train_losses = 0

            for iteration in range(iter_no):  # iter_no=13  一共有13个batch, 每个batch有8个样本

                
                self.D.zero_grad()
                given_x = pop_dec_aug[iteration * self.batchsize: (1 + iteration) * self.batchsize, :]   # 一个batchsize的解   given_x=(8, 31)
                given_y = labels_aug[iteration * self.batchsize: (1 + iteration) * self.batchsize]   # 对应的一个batchsize的label  given_y=(8,1)
                batch_size = np.shape(given_x)[0]  # 因为最后一个batch可能没有8个，所以这里要记录一下batch_size的大小

                given_x_ = Variable(torch.from_numpy(given_x).cpu()).float()
                given_y = Variable(torch.from_numpy(given_y).cpu()).float()
                # given_x_ = Variable(torch.from_numpy(given_x).to(device)).float()
                # given_y = Variable(torch.from_numpy(given_y).to(device)).float()
                # 在Pytorch0.4.0及以后，Tensor和Variable已经合并
                # given_x_ = torch.from_numpy(given_x).to(device).float()   # numpy->tensor   (8, 31)
                # given_y = torch.from_numpy(given_y).to(device).float()    # （8，1）
                # 注意上面的given_x_, given_y都是真实的数据
                # d_results_real = self.D(given_x_.detach())   # 这里应该是不需要detach操作，因为given_x_不是可学习的参数
                d_results_real = self.D(given_x_)   # xwf   

                # 这里的fake_x就是噪声，将fake_x经过G来生成假的数据, fake_y都是random出来的数据
                fake_x = np.random.multivariate_normal(center, cov, batch_size)  # （8， 31）从噪声出发
                fake_x = torch.from_numpy(np.maximum(np.minimum(fake_x, np.ones((batch_size, self.d))),
                                                         np.zeros((batch_size, self.d))))

                fake_y = Variable(torch.zeros((batch_size, 1)).cpu())
                fake_x_ = Variable(fake_x.cpu()).float()
                # fake_y = torch.zeros((batch_size, 1)).to(device)   # 因为是假的数据嘛，所以fake_y都是0
                # fake_x_ = fake_x.to(device).float()

                # g_results = self.G(fake_x_.detach())  # g_results=(8,31)   这里写错了，感觉应该是g_results=self.G(fake_x_)    d_results_fake=self.D(g_results.detach)
                # d_results_fake = self.D(g_results)  # 因为这里通过g_results会涉及到G的更新，如果这里也设置g_results，则无法梯度回传去更新G
                g_results = self.G(fake_x_)          # xwf
                d_results_fake = self.D(g_results.detach())

                d_train_loss = self.BCE_loss(d_results_real, given_y) + \
                               self.BCE_loss(d_results_fake, fake_y)  
                d_train_loss.backward()
                self.D_optimizer.step()

                
                self.G.zero_grad()
                fake_x = np.random.multivariate_normal(center, cov, batch_size)
                fake_x = torch.from_numpy(np.maximum(np.minimum(fake_x, np.ones((batch_size, self.d))),
                                                     np.zeros((batch_size, self.d))))
                fake_x_ = Variable(fake_x.cpu()).float()
                fake_y = Variable(torch.ones((batch_size, 1)).cpu())
                # fake_x_ = fake_x.to(device).float()
                # fake_y = torch.ones((batch_size, 1)).to(device)  # 这里你希望G生成的内容经过判别器后能够尽可能地接近1，说明生成的就越真实
                g_results = self.G(fake_x_)
                d_results = self.D(g_results)
                g_train_loss = self.BCE_loss(d_results, fake_y)   
                g_train_loss.backward()
                self.G_optimizer.step()
                g_train_losses += g_train_loss.cpu()
                # g_train_losses += g_train_loss.item()

            # print("Epoch[{}], loss: {:.5f}".format(epoch, g_train_losses))

            random.shuffle(indices)
            pop_dec_aug = pop_dec_aug[indices, :]   # 感觉这里应该加上label = labels[indices, :]
            labels_aug = labels_aug[indices, :]   #  xwf

    def generate(self, sample_noises, population_size):  # sample_noises.shape=(10, 31)  population_size=100

        self.G.eval()  

        center = np.mean(sample_noises, axis=0).T   # shape=(31,)
        cov = np.cov(sample_noises.T)   # (31, 31)
        batch_size = population_size    # bs = 100

        noises = np.random.multivariate_normal(center, cov, batch_size)   # (100, 31)
        noises = torch.from_numpy(np.maximum(np.minimum(noises, np.ones((batch_size, self.d))),
                                                      np.zeros((batch_size, self.d))))
        # noises = noises.to(device).float() # 数据移到GPU    (batchsize,n_sample)=(100, 31)   一个batch里面有31个样本，就要预测31个结果，这里有100个batchsize
        # with torch.no_grad(): #关闭autograd
        #     decs = self.G(noises).cpu().data.numpy() # 生成结果并转回CPU     shape=(100, 31)
        decs = self.G(Variable(noises.cpu()).float()).cpu().data.numpy()
        return decs

    def discrimate(self, off):

        self.D.eval()  
        batch_size = off.shape[0]
        off = off.reshape(batch_size, 1, off.shape[1])
        
        x = Variable(torch.from_numpy(off).cpu(), volatile=True).float()
        d_results = self.D(x).cpu().data.numpy()
        # with torch.no_grad():
        #     x = torch.from_numpy(off).to(device).float()
        #     d_results = self.D(x).cpu().data.numpy()

        return d_results.reshape(batch_size)



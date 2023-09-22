import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import random
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Generator(nn.Module):
    
    def __init__(self, d, n_noise):  # 1-d vector   d=31, n_noise=31
        super(Generator, self).__init__()
        self.linear1 = nn.Linear(n_noise, d, bias=True)
        self.bn1 = nn.BatchNorm1d(d)
        self.linear2 = nn.Linear(d, d, bias=True)
        self.bn2 = nn.BatchNorm1d(d)
        self.linear3 = nn.Linear(d, d, bias=True)
        self.bn3 = nn.BatchNorm1d(d)

    
    def forward(self, noise):
        x = torch.tanh(self.bn1(self.linear1(noise)))
        x = torch.tanh(self.bn2(self.linear2(x)))
        x = torch.sigmoid(self.bn3(self.linear3(x)))  
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


class GAN(object):# d=31, batchsize=8, lr=0.001, epoches=200, n_noise=31
    def __init__(self, d, batchsize, lr, epoches, n_noise):   
        self.d = d
        self.n_noise = n_noise
        self.BCE_loss = nn.BCELoss()
        self.G = Generator(self.d, self.n_noise).to(device)
        self.D = Discriminator(self.d).to(device)
        # self.G.cpu()
        # self.D.cpu()
        self.G_optimizer = optim.Adam(self.G.parameters(), 4*lr)
        self.D_optimizer = optim.Adam(self.D.parameters(), lr)
        self.epoches = epoches
        self.batchsize = batchsize

    def train(self, pop_dec, labels, samples_pool):  # pop_dec.shape=(100, 31), labels.shape=(100, 1), samples_pool.shape=(10, 31)
        self.D.train()
        self.G.train()
        n, d = np.shape(pop_dec)  # n=100,  d=31
        indices = np.arange(n)  # indices=array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,..., 98, 99])
        
        center = np.mean(samples_pool, axis=0)  # (31,1)  axis=0，对第一个维度求均值    下面的 cov 矩阵提供了一个关于这10个样本在31个特征上相互关系的全面视图。
        cov = np.cov(samples_pool[:10, :].reshape((d, samples_pool[:10, :].size // d)))#  (10, 31)->(31, 10)  conv=(31,31)  np.cov 函数用于计算协方差矩阵   samples_pool.shape=(10, 31),   
        iter_no = (n + self.batchsize - 1) // self.batchsize  # batchsize=8   n=100  iter_no:代表着在给定的设置中，需要多少批（batch）迭代来处理所有 n 个样本。其中，每个批的大小由 self.batchsize（在这个例子中是8）确定。

        for epoch in range(self.epoches): # epoches=200
            g_train_losses = 0

            for iteration in range(iter_no):  # iter_no=13  一共有13个batch, 每个batch有8个样本

                
                self.D.zero_grad()
                given_x = pop_dec[iteration * self.batchsize: (1 + iteration) * self.batchsize, :]   # 一个batchsize的解   given_x=(8, 31)
                given_y = labels[iteration * self.batchsize: (1 + iteration) * self.batchsize]   # 对应的一个batchsize的label  given_y=(8,1)
                batch_size = np.shape(given_x)[0]  # 因为最后一个batch可能没有8个，所以这里要记录一下batch_size的大小

                # given_x_ = Variable(torch.from_numpy(given_x).cpu()).float()
                # given_y = Variable(torch.from_numpy(given_y).cpu()).float()
                # given_x_ = Variable(torch.from_numpy(given_x).to(device)).float()
                # given_y = Variable(torch.from_numpy(given_y).to(device)).float()
                # 在Pytorch0.4.0及以后，Tensor和Variable已经合并
                given_x_ = torch.from_numpy(given_x).to(device).float()   # numpy->tensor   (8, 31)
                given_y = torch.from_numpy(given_y).to(device).float()    # （8，1）

                d_results_real = self.D(given_x_.detach())

            
                fake_x = np.random.multivariate_normal(center, cov, batch_size)
                fake_x = torch.from_numpy(np.maximum(np.minimum(fake_x, np.ones((batch_size, self.d))),
                                                         np.zeros((batch_size, self.d))))

                # fake_y = Variable(torch.zeros((batch_size, 1)).cpu())
                # fake_x_ = Variable(fake_x.cpu()).float()
                fake_y = torch.zeros((batch_size, 1)).to(device)
                fake_x_ = fake_x.to(device).float()

                g_results = self.G(fake_x_.detach())
                d_results_fake = self.D(g_results)

                d_train_loss = self.BCE_loss(d_results_real, given_y) + \
                               self.BCE_loss(d_results_fake, fake_y)  
                d_train_loss.backward()
                self.D_optimizer.step()

                
                self.G.zero_grad()
                fake_x = np.random.multivariate_normal(center, cov, batch_size)
                fake_x = torch.from_numpy(np.maximum(np.minimum(fake_x, np.ones((batch_size, self.d))),
                                                     np.zeros((batch_size, self.d))))
                # fake_x_ = Variable(fake_x.cpu()).float()
                # fake_y = Variable(torch.ones((batch_size, 1)).cpu())
                fake_x_ = fake_x.to(device).float()
                fake_y = torch.ones((batch_size, 1)).to(device)
                g_results = self.G(fake_x_)
                d_results = self.D(g_results)
                g_train_loss = self.BCE_loss(d_results, fake_y)   
                g_train_loss.backward()
                self.G_optimizer.step()
                # g_train_losses += g_train_loss.cpu()
                g_train_losses += g_train_loss.item()
            
            random.shuffle(indices)
            pop_dec = pop_dec[indices, :]

    def generate(self, sample_noises, population_size):

        self.G.eval()  

        center = np.mean(sample_noises, axis=0).T   # shape=(31,)
        cov = np.cov(sample_noises.T)   # (31, 31)
        batch_size = population_size    # bs = 100

        noises = np.random.multivariate_normal(center, cov, batch_size)   # (100, 31)
        noises = torch.from_numpy(np.maximum(np.minimum(noises, np.ones((batch_size, self.d))),
                                                      np.zeros((batch_size, self.d))))
        noises = noises.to(device).float() # 数据移到GPU    (batchsize,n_sample)=(100, 31)   一个batch里面有31个样本，就要预测31个结果，这里有100个batchsize
        with torch.no_grad(): #关闭autograd
            # decs = self.G(Variable(noises.cpu()).float()).cpu().data.numpy()
            decs = self.G(noises).cpu().data.numpy() # 生成结果并转回CPU     shape=(100, 31)
        return decs

    def discrimate(self, off):

        self.D.eval()  
        batch_size = off.shape[0]
        off = off.reshape(batch_size, 1, off.shape[1])
        
        # x = Variable(torch.from_numpy(off).cpu(), volatile=True).float()
        # d_results = self.D(x).cpu().data.numpy()
        with torch.no_grad():
            x = torch.from_numpy(off).to(device).float()
            d_results = self.D(x).cpu().data.numpy()

        return d_results.reshape(batch_size)



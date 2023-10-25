import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import random
import numpy as np
import torch.nn.functional as F


class Generator(nn.Module):
    
    def __init__(self, d, n_noise):  # 1-d vector
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
        # x = torch.sigmoid(self.bn3(self.linear3(x)))
        x = F.softmax(self.bn3(self.linear3(x)), dim=1)  # 使用softmax确保输出的每个维度的和为1
        return x


class Discriminator(nn.Module):
    
    def __init__(self, d):
        super(Discriminator, self).__init__()
        self.linear1 = nn.Linear(d, d, bias=True)
        self.linear2 = nn.Linear(d, 1, bias=True)

    
    def forward(self, dec):
        x = torch.tanh(self.linear1(dec))
        x = torch.sigmoid(self.linear2(x))
        return x


class GAN(object):
    def __init__(self, d, batchsize, lr, epoches, n_noise):
        self.d = d
        self.n_noise = n_noise
        self.BCE_loss = nn.BCELoss()
        self.G = Generator(self.d, self.n_noise)
        self.D = Discriminator(self.d)
        self.G.cpu()
        self.D.cpu()
        self.G_optimizer = optim.Adam(self.G.parameters(), 4*lr)
        self.D_optimizer = optim.Adam(self.D.parameters(), lr)
        self.epoches = epoches
        self.batchsize = batchsize

    def train(self, pop_dec, labels, samples_pool):
        self.D.train()
        self.G.train()
        n, d = np.shape(pop_dec)
        indices = np.arange(n)

        center = np.mean(samples_pool, axis=0)
        cov = np.cov(samples_pool[:10, :].reshape((d, samples_pool[:10, :].size // d)))
        iter_no = (n + self.batchsize - 1) // self.batchsize

        for epoch in range(self.epoches):
            g_train_losses = 0

            for iteration in range(iter_no):

                
                self.D.zero_grad()
                given_x = pop_dec[iteration * self.batchsize: (1 + iteration) * self.batchsize, :]
                given_y = labels[iteration * self.batchsize: (1 + iteration) * self.batchsize]
                batch_size = np.shape(given_x)[0]

                
                given_x_ = Variable(torch.from_numpy(given_x).cpu()).float()
                given_y = Variable(torch.from_numpy(given_y).cpu()).float()
                d_results_real = self.D(given_x_.detach())

            
                fake_x = np.random.multivariate_normal(center, cov, batch_size)
                fake_x = torch.from_numpy(np.maximum(np.minimum(fake_x, np.ones((batch_size, self.d))),
                                                         np.zeros((batch_size, self.d))))

                fake_y = Variable(torch.zeros((batch_size, 1)).cpu())
                fake_x_ = Variable(fake_x.cpu()).float()
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
                fake_x_ = Variable(fake_x.cpu()).float()
                fake_y = Variable(torch.ones((batch_size, 1)).cpu())
                g_results = self.G(fake_x_)
                d_results = self.D(g_results)
                g_train_loss = self.BCE_loss(d_results, fake_y)   
                g_train_loss.backward()
                self.G_optimizer.step()
                g_train_losses += g_train_loss.cpu()
            
            random.shuffle(indices)
            pop_dec = pop_dec[indices, :]

    def generate(self, sample_noises, population_size):

        self.G.eval()  

        center = np.mean(sample_noises, axis=0).T  
        cov = np.cov(sample_noises.T)   
        batch_size = population_size

        noises = np.random.multivariate_normal(center, cov, batch_size)
        noises = torch.from_numpy(np.maximum(np.minimum(noises, np.ones((batch_size, self.d))),
                                                      np.zeros((batch_size, self.d))))
        decs = self.G(Variable(noises.cpu()).float()).cpu().data.numpy()
        return decs

    def discrimate(self, off):

        self.D.eval()  
        batch_size = off.shape[0]
        off = off.reshape(batch_size, 1, off.shape[1])

        x = Variable(torch.from_numpy(off).cpu(), volatile=True).float()
        d_results = self.D(x).cpu().data.numpy()
        return d_results.reshape(batch_size)



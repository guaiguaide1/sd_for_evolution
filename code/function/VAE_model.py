import torch
import torch.nn as nn
import torch.optim as optim
import random 
import numpy as np 
from sklearn.utils import resample

# 定义模型
class VAENet(nn.Module):
    def __init__(self, input_dim=784):
        super(VAENet, self).__init__()
        self.dim = input_dim
        hidden_dim = self.dim
        latent_dim = self.dim 
        # 编码器
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # 均值
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # 方差

        # 解码器
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar






class VAE(object):
    def __init__(self, dim, lr, epoches):
        # 1. 实例化参数
        self.dim = dim 
        self.lr = lr 
        self.epoches = epoches 

        # 2. 实例化模型
        self.model = VAENet(self.dim)

        # 3. 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
    

    def loss_func(self, recon_x, x, mu, logvar, negative_samples):
        # 重构误差
        MSE = torch.mean((recon_x - x)**2)

        # KL散度
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # loss3: 确保生成的每个正样本都尽可能地远离所有负样本
        # 先计算了每个正样本与所有负样本之间的欧氏距离，然后选择了每个正样本的最差匹配
        # 计算欧氏距离
        distances = torch.norm(recon_x.unsqueeze(1) - negative_samples, dim=2)
        # 选择每个正样本的最差匹配（距离最大的负样本）
        worst_matches = torch.max(distances, dim=1)[0]
        # 定义一个损失函数来最大化最差匹配的距离
        FarAwayN =  -torch.mean(worst_matches)

        # loss3: 计算两两之间的距离
        distances = torch.cdist(recon_x, recon_x)
        # 将对角线上的值(每个解与自身的距离)设置为0
        mask = torch.eye(len(recon_x)) == 1
        distances.masked_fill(mask, 0)
        #计算每个解的多样性度量，即与其他解的平均距离
        diversity_measures = distances.sum(1) / (len(recon_x) - 1)
        # 计算整体的多样性度量
        Diversity = diversity_measures.mean()

        # print("MSE:", MSE.shape)
        # print("KLD:", KLD.shape)
        # print("FarAwarN:", FarAwayN.shape)
        # print("Diversity:", Diversity.shape)

        # return MSE + KLD + FarAwayN + Diversity
        return MSE+KLD

    def train(self, pop_dec, positive_samples, negative_samples):
        self.model.train()

        # 定义目标样本数量
        target_samples = 50
        
        # # # 过采样正样本至50个
        upsampled_positive_samples = resample(positive_samples, 
                                            replace=True, 
                                            n_samples=target_samples,
                                            random_state=123)

        data = torch.from_numpy(upsampled_positive_samples).float()
        negative_samples = torch.from_numpy(negative_samples).float()
        for epoch in range(self.epoches):
            total_loss = 0 
            self.optimizer.zero_grad()
            recon_data, mu, logvar = self.model(data)

            loss = self.loss_func(recon_data, data, mu, logvar, negative_samples)            
            loss.backward()
            total_loss = loss.item()
            self.optimizer.step()
            # print("Epoch[{}], loss: {:.5f}".format(epoch, total_loss))
    
    def generate(self, population_size):
        self.model.eval()

        z = torch.randn(population_size, self.dim)
        with torch.no_grad():
            samples = self.model.decode(z).cpu().data.numpy()
        return samples

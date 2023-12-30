
import torch
from torch import nn

import numpy as np



class GaussianMixture(nn.Module):
    def __init__(self, input_size, hidden_size, num_gaussians):
        super(GaussianMixture, self).__init__()
        
        self.z_h = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh()
        )
        
        self.z_pi = nn.Linear(hidden_size, num_gaussians)
        self.z_sigma = nn.Linear(hidden_size, num_gaussians)
        self.z_mu = nn.Linear(hidden_size, num_gaussians)  

    def forward(self, x):
        z_h = self.z_h(x)
        pi = nn.functional.softmax(self.z_pi(z_h), -1)
        sigma = torch.exp(self.z_sigma(z_h))
        mu = self.z_mu(z_h)
        return pi, sigma, mu

def gaussian_distribution(y, mu, sigma):
    # make |mu|=K copies of y, subtract mu, divide by sigma
    result = (y.expand_as(mu) - mu) * torch.reciprocal(sigma)
    result = -0.5 * (result * result)
    result = (torch.exp(result) * torch.reciprocal(sigma)) * (1.0/np.sqrt(2.0*np.pi))
    return result

def gaussion_loss(pi, sigma, mu, y):
    result = gaussian_distribution(y, mu, sigma) * pi
    result = torch.sum(result, dim=1)
    result = -torch.log(result)
    return torch.mean(result)


# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt # creating visualizations
import numpy as np # basic math and random numbers
import torch # package for building functions with learnable parameters
import torch.nn as nn # prebuilt functions specific to neural networks
from torch.autograd import Variable # storing data while learning
def generate_data(n_samples):
    epsilon = np.random.normal(size=(n_samples))
    x_data = np.random.uniform(-50, 0, n_samples)
    y_data = 7*np.sin(0.75*x_data) + 0.5*x_data + epsilon
    return x_data, y_data
    
n_samples = 1000
x_data, y_data = generate_data(n_samples)

plt.figure(figsize=(8, 8))
plt.scatter(x_data, y_data, alpha=0.2)
plt.show()

n_input = 1
n_hidden = 20
n_output = 1
x_tensor = torch.from_numpy(np.float32(x_data).reshape(n_samples, n_input))
y_tensor = torch.from_numpy(np.float32(y_data).reshape(n_samples, n_input))
x_variable = Variable(x_tensor)
y_variable = Variable(y_tensor, requires_grad=False)
x_test_data = np.linspace(-50, 0, n_samples)

# change data shape, move from numpy to torch
x_test_tensor = torch.from_numpy(np.float32(x_test_data).reshape(n_samples, n_input))
x_test_variable = Variable(x_test_tensor)
x_variable.data = y_tensor
y_variable.data = x_tensor
x_test_data = np.linspace(0, 50, n_samples)
x_test_tensor = torch.from_numpy(np.float32(x_test_data).reshape(n_samples, n_input))
x_test_variable.data = x_test_tensor

mdn_x_data = y_data
mdn_y_data = x_data
mdn_x_tensor = y_tensor
mdn_y_tensor = x_tensor
x_variable = mdn_x_tensor
y_variable = mdn_y_tensor

network = GaussianMixture(1, 20,5)
optimizer = torch.optim.Adam(network.parameters())


for epoch in range(10000):
    pi_variable, sigma_variable, mu_variable = network(x_variable)
    loss = gaussion_loss(pi_variable, sigma_variable, mu_variable, y_variable)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
        
    if epoch % 500 == 0:
        print(epoch, loss)

pi_variable, sigma_variable, mu_variable = network(x_test_variable)
pi_data = pi_variable.data.numpy()
sigma_data = sigma_variable.data.numpy()
mu_data = mu_variable.data.numpy()

def gumbel_sample(x, axis=1):
    z = np.random.gumbel(loc=0, scale=1, size=x.shape)
    return (np.log(x) + z).argmax(axis=axis)

k = gumbel_sample(pi_data)

indices = (np.arange(n_samples), k)
rn = np.random.randn(n_samples)
sampled = rn * sigma_data[indices] + mu_data[indices]

plt.figure(figsize=(8, 8))
plt.scatter(mdn_x_data, mdn_y_data, alpha=0.2)
plt.scatter(x_test_data, sampled, alpha=0.2, color='red')
plt.show()




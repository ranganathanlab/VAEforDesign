#!/usr/bin/env python
# coding: utf-8

#By: Xinran Lian, Andrew Ferguson

# VAE examples  
# https://chrisorm.github.io/VAE-pyt.html  
# https://github.com/pytorch/examples/tree/master/vae  

# https://www.quora.com/Can-l-combine-dropout-and-l2-regularization

import sys
import numpy as np
import argparse
import os
import time
import pickle
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib as mpl

class VAE(nn.Module):
    def __init__(self, q, d):
        super(VAE, self).__init__()
        self.hsize=int(1.5*q) # size of hidden layer
        
        self.en1 = nn.Linear(q, self.hsize)
        self.en2 = nn.Linear(self.hsize, self.hsize) #
        self.en3 = nn.Linear(self.hsize, self.hsize)
        self.en_mu = nn.Linear(self.hsize, d)
        self.en_std = nn.Linear(self.hsize, d) # Is it logvar?
        
        self.de1 = nn.Linear(d, self.hsize)
        self.de2 = nn.Linear(self.hsize, self.hsize) #
        self.de22 = nn.Linear(self.hsize, self.hsize)
        self.de3 = nn.Linear(self.hsize, q)     
 
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()        
        self.softmax = nn.Softmax(dim=1)
        
        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)
        
        self.bn1 = nn.BatchNorm1d(self.hsize) # batchnorm layer
        self.bn2 = nn.BatchNorm1d(self.hsize)
        self.bn3 = nn.BatchNorm1d(self.hsize)
        self.bnfinal = nn.BatchNorm1d(q)  

    def encode(self, x):
        """Encode a batch of samples, and return posterior parameters for each point."""
        x = self.tanh(self.en1(x)) # first encode
        x = self.dropout1(x) 
        x = self.tanh(self.en2(x))
        x = self.bn1(x)
        x = self.tanh(self.en3(x)) # second encode
        return self.en_mu(x), self.en_std(x) # third (final) encode, return mean and variance
    
    def decode(self, z):
        """Decode a batch of latent variables"""
        z = self.tanh(self.de1(z))
        z = self.bn2(z)
        z = self.tanh(self.de2(z))
        z = self.dropout2(z)
        z = self.tanh(self.de22(z))
        
        # residue-based softmax
        # - activations for each residue in each position ARE constrained 0-1 and ARE normalized (i.e., sum_q p_q = 1)
        z = self.bn3(z)
        z = self.de3(z)
        z = self.bnfinal(z)
        z_normed = torch.FloatTensor() # empty tensor?
        z_normed = z_normed.to(device) # store this tensor in GPU/CPU
        for j in range(n):
            start = np.sum(q_n[:j])
            end = np.sum(q_n[:j+1])
            z_normed_j = self.softmax(z[:,start:end])
            z_normed = torch.cat((z_normed,z_normed_j),1)
        return z_normed
    
    def reparam(self, mu, logvar): 
        """Reparameterisation trick to sample z values. 
        This is stochastic during training, and returns the mode during evaluation.
        Reparameterisation solves the problem of random sampling is not continuous, which is necessary for gradient descent
        """
        if self.training:
            std = logvar.mul(0.5).exp_() 
            eps = std.data.new(std.size()).normal_() # torch variable
            return eps.mul(std).add_(mu)
        else:
            return mu      
    
    def forward(self, x):
        """Takes a batch of samples, encodes them, and then decodes them again to compare."""
        mu, logvar = self.encode(x.view(-1, q)) # get mean and variance
        z = self.reparam(mu, logvar)
        return self.decode(z), mu, logvar
    
    def loss(self, reconstruction, x, mu, logvar): 
        """ELBO assuming entries of x are binary variables, with closed form KLD."""
        bce = torch.nn.functional.binary_cross_entropy(reconstruction, x.view(-1, q))
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Normalise by same number of elements as in reconstruction
        KLD /= x.view(-1, q).data.shape[0] * q 
        return bce + KLD
    
    def get_z(self, x):
        """Encode a batch of data points, x, into their z representations."""
        mu, logvar = self.encode(x.view(-1, q))
        return self.reparam(mu, logvar)
    
    
def VAEtrain(model, epoch, batches_per_epoch, v_train, v_val):
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    
    # batching and training
    ind = np.arange(v_train.shape[0])
    for i in range(batches_per_epoch):
        data = torch.FloatTensor(v_train[np.random.choice(ind, size=batch_size)]) # randomly sample training set
        data = data.to(device)
        optimizer.zero_grad()
        pred, mu, logvar = model(data)
        loss = model.loss(pred, data, mu, logvar) #loss(self, reconstruction, x, mu, logvar)
        loss.backward()
        optimizer.step() # optimize function...
    
    # training loss
    data = torch.FloatTensor(v_train)
    data = data.to(device)
    pred, mu, logvar = model(data)
    train_loss = model.loss(pred, data, mu, logvar)
    train_loss = train_loss.cpu().detach().numpy() # network is trained on this loss, maximize P(X), what the network see
    
    diff = pred.cpu().detach().numpy() - v_train
    train_loss_MSE = np.mean(diff**2) # mean square error per position, for human to see. Network does not know it.
    
    # validation loss
    data = torch.FloatTensor(v_val)
    data = data.to(device)
    pred, mu, logvar = model(data)
    val_loss = model.loss(pred, data, mu, logvar)
    val_loss = val_loss.cpu().detach().numpy()
    
    diff = pred.cpu().detach().numpy() - v_val
    val_loss_MSE = np.mean(diff**2)
    
    if (epoch % 10 == 0):
        print('====> Epoch %d done! Train loss = %.2e, Val loss = %.2e, Train loss MSE = %.2e, Val loss MSE = %.2e' % (epoch,train_loss,val_loss,train_loss_MSE,val_loss_MSE))
    
    return train_loss, val_loss, train_loss_MSE, val_loss_MSE


def VAEtest(model, v_test):
    
    data = torch.FloatTensor(v_test)
    data = data.to(device)
    pred, mu, logvar = model(data) # model is VAE?
    
    # ELBO test loss
    test_loss = model.loss(pred, data, mu, logvar)
    test_loss = test_loss.cpu().detach().numpy()
    
    # MSE test loss
    diff = pred.cpu().detach().numpy() - v_test
    test_loss_MSE = np.mean(diff**2)
    
    return pred, test_loss, test_loss_MSE
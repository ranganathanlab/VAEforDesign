#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import os
from numba import jit
import argparse
import pickle
import multiprocessing as mp
import time
import toolkit
import shutil

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.multiprocessing import Pool, Process, set_start_method
from sklearn.model_selection import train_test_split
from torch.autograd import Variable

from Bio import SeqIO
from Bio import AlignIO

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

        #replace tanh with relu
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
            eps = std.data.new(std.size()).normal_() # normal distribution
            return eps.mul(std).add_(mu)
        else:
            return mu      
    
    def forward(self, x):
        """Takes a batch of samples, encodes them, and then decodes them again to compare."""
        mu, logvar = self.encode(x.view(-1, q)) # get mean and variance
        z = self.reparam(mu, logvar) # sampling latent variable z from mu and logvar
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

def generate(seed, n, n_gen, n_samp):
    np.random.seed(seed)
    z_gen = np.random.normal(0., 1., (n_gen, d)) #generate normal distribution of random numbers
    data = torch.FloatTensor(z_gen).to(device)
    data = model.decode(data) # Use the decoding layer to generate new sequences.
    v_gen = data.cpu().detach().numpy()
    sample_list = []

    for i in range(n_gen):
        for k in range(n_samp):
            v_samp_nothot = toolkit.sample_seq(seed+k, q, n, q_n, i, v_gen)
            sample_list.append([v_samp_nothot,z_gen[i]])
    return sample_list

def run_gen(s): # for parallelization
    return generate(s, n, int(n_gen/10), n_sample)
    
if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='Hint: In total ngen*nsamp new sequences are generated, default 1000. Then they are filtered according to thresholds of minimum Hamming distance.')
    
    parser.add_argument("-g", "--ngen", dest ="ngen", 
                        default=1000, type=int, 
                        help="times of sampling in the latent space. Default 1000. Recommended to enter a multiple of 10.")
    parser.add_argument("-s", "--nsamp", dest ="nsamp", 
                        default=10, type=int, 
                        help="times of throwing dice at each sampling point. Default 10")
    parser.add_argument("-r", "--randseed", dest ="randseed", 
                        default=1000, type=int, help="Random seed. Default 1000.")
    parser.add_argument("-n", "--name", dest ="name", 
                        default='protein', type=str, help="Name of your protein.")
    parser.add_argument("-c", "--custom", dest ="custom", 
                        default='', type=str, 
                        help="A custom string for your generated sequence file name. Default None.")
    
    parser.add_argument("-a", "--sca", dest="sca", action="store_true",
                        default=False, help="Compute SCA for generated sequecnes")
    
    options = parser.parse_args()
    device = torch.device("cpu")
    torch.manual_seed(20)

    # Load data
    path = '../Outputs/'
    parameters = pickle.load(open(path + options.name + ".db", 'rb'))
    q_n = parameters['q_n']
    aaindex = parameters['index']
    v_traj_onehot = parameters['onehot']
    records_MSA = parameters['seq']

    N=np.size(v_traj_onehot,axis=0)
    q=np.size(v_traj_onehot,axis=1)
    n=np.size(q_n)
    
    # load VAE
    d=3
    model = VAE(q,d)
    model.load_state_dict(torch.load('VAE.pyt',map_location='cpu'))
    model.eval()
        
    # Generate new sequences
    start_all = time.time()
    
    seed = options.randseed
    n_gen = options.ngen
    n_sample = options.nsamp
    
    np.random.seed(seed)
    real_nohot_list = toolkit.convert_nohot(v_traj_onehot, q_n)
    seed_list = np.random.randint(0, 2**32, 10)
    pool = mp.Pool(mp.cpu_count())
    
    print('Start generating sequences...')
    st_time = time.time()
    packed = pool.map(run_gen, seed_list)
    packed = [item for sublist in packed for item in sublist]
    sample_list = [i[0] for i in packed]
    z_list = [i[1] for i in packed]
    # https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
    pool.close()
    alp_new_seq = toolkit.convert_alphabet(np.array(sample_list), aaindex, q_n) 
    end_time = time.time()
    print("Elapsed time %.2f (s)" % (end_time - st_time))
    
    print('Computing VAE logP for selected sequences...')
    st_time = time.time()
    
    print('Converting generated sequences to Potts...')
    new_potts, _ = toolkit.convert_potts(alp_new_seq, aaindex)
    print('Reconstructing with VAE...')
    pred_ref,_,_ = model(torch.FloatTensor(new_potts))
    p_weight = pred_ref.cpu().detach().numpy()
    print('computing logP...')
    log_norm = toolkit.make_logP(new_potts, p_weight, q_n)
    
    if options.sca:
        print('Start computing SCA...')
        filename = options.name+ options.custom+'_sca'
        if os.path.isdir('output')==0:
            os.mkdir('output')
        with open('output/' + filename+'.fasta', 'w') as f:
            f.write(">2vkn_chainA_p001\n")
            f.write("NFIYKAKALYPYDADDAYEISFEQNEILQVSDIEGRWWKARRNGETGIIPSNYVQLIDG\n") #2vkn_chainA_p001
            for item in alp_new_seq[:-1]:
                f.write(">gi\n")
                f.write("%s\n" % item)
        os.system('scaProcessMSA output/' + filename +'.fasta -s 2VKN -c A -o 0')
        os.system('scaCore output/' + filename +'.db')
        os.system('scaSectorID output/' + filename +'.db')
        
        if os.path.isfile(path + filename +'.db'):
            os.remove(path + filename +'.db')
        os.rename('output/'+filename +'.db',path + filename +'.db')
        shutil.rmtree('output')
        print('SCA computing finished.')
    
    end_time = time.time()
    print("Elapsed time %.2f (s)" % (end_time - st_time))
    
    np.savez(path + options.name + options.custom + 'gen_data.npz', seq = alp_new_seq, ham = 0, logP = log_norm, z_list = z_list)
    
    end_all = time.time()
    print("\nTotal elapsed time %.2f (s)" % (end_all - start_all))
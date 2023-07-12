'''
By: 
Xinran Lian, Andrew Ferguson

'''

from __future__ import division
import numpy as np
import pandas as pd
import os
import pickle
from numba import jit
import time

from scipy.special import kl_div
from scipy.spatial import distance

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

from sklearn.model_selection import train_test_split
from Bio import SeqIO

##################### Pretreatment #####################

# Read the fasta file
def get_seq(filename, get_header = False):
    assert filename.endswith('.fasta'), 'Not a fasta file.'
    
    records = list(SeqIO.parse(filename, "fasta"))
    records_seq = [i.seq for i in records]
    headers = [i.description for i in records]
    if get_header == True:
        return records_seq, headers
    else:
        return records_seq

# Output indices of AAs at each position
def potts_index(sequence): # Output all possible AAs on all positions
    for i in range(len(sequence)):
        assert len(sequence[i]) == len(sequence[0])
        
        aa_at_pos = []
        for i in range(len(sequence[0])):
            tmp_list = []
            for j in range(len(sequence)):
                if (sequence[j][i] in tmp_list) == False:
                    tmp_list.append(sequence[j][i])
            aa_at_pos.append(tmp_list)
        return [''.join(i) for i in aa_at_pos]
    
# Convert alphabet sequences to one-hot Potts representation.
def convert_potts(sequence, aa_at_pos):
    N = len(sequence) # samples
    n = len(aa_at_pos) # positions
    q_n = np.array([len(i) for i in aa_at_pos]).astype(int)
    length = sum(q_n)

    v_traj_onehot = np.zeros((N,length)).astype(int)
    for j in range(n):
        sumq = sum(q_n[:j])
        for i in range(N):
            pos_to_1 = sumq + aa_at_pos[j].find(sequence[i][j])
            v_traj_onehot[i,pos_to_1] = 1 # set a position as 1
    return v_traj_onehot, q_n


##################### Treatments after training the model #####################
'''
Warning: 
convert_nohot and aa2int: Both functions convert alphabet representation of AAs to int representation. 
aa2int is only used for plmDCA energy. In other cases we use convert_nohot.

Difference between the two functions is they use different dictionary: 
*aa2int* uses *plm_dict.dict_int2aa*, where all 21 AAs (including gap) is correspond to an int number from 0 to 20.
*convert_nohot* uses *aa_at_pos*. This index is based on possibilities of AAs at each position.  AAs not appearing at a position in the whole MSA will not be in the index. This index is designed for the VAE model, to avoid all-zero features.
'''

# Convert alphabet MSA to int representation:
def convert_nohot(sequence_list, q_n):
    sequence_list = sequence_list.reshape([-1,sum(q_n)])
    length = np.size(sequence_list,axis=0)
    real_nothot = np.zeros([length,len(q_n)])
                     
    for i in range(length):
        tmp_seq = sequence_list[i]
        for j in range(len(q_n)):
            start = np.sum(q_n[:j])
            end = np.sum(q_n[:j+1])
            real_nothot[i,j] = np.argmax(tmp_seq[start:end])
    return real_nothot

# Convert int sequences to alphabet.
def convert_alphabet(int_seq, aaindex, q_n):
    seqs_alp=[]
    int_seq = int_seq.reshape([-1,len(q_n)])
    for i in range(len(int_seq)):
        tmpseq=''
        for pos in range(len(int_seq[0])):
            num=int(int_seq[i][pos])
            tmpseq+=aaindex[pos][num]
        #tmpseq=np.array(tmpseq)
        seqs_alp.append(tmpseq)
    return(seqs_alp)

# Make an m × n matrix (arXiv:1712.03346)
# m is the number of amino-acids, n the number of positions considered in the protein alignment.
@jit(nopython=True)
def make_matrix(sequence, n, q_n): # a sequence of interest
    matrix = np.zeros((21,n),np.float64)
    for j in range(n):
        start = np.sum(q_n[:j])
        end = np.sum(q_n[:j+1])
        matrix[:q_n[j],j] = sequence[start:end]
    return matrix

def make_logP(new_potts, p_weight,q_n):
    n = len(q_n)
    return [np.log(np.trace(make_matrix(new_potts[i],n,q_n).T @ make_matrix(p_weight[i],n,q_n))) for i in range(len(new_potts))] - np.log(n)

##################### Generate and analyze new sequences #####################
class plm_dict:
    dict_int2aa = {0:"A",1:"C",2:"D",3:"E",4:"F",5:"G",6:"H",7:"I",8:"K",9:"L",10:"M",11:"N",12:"P",13:"Q",14:"R",15:"S",16:"T",17:"V",18:"W",19:"Y",20:"X"}
    dict_aa2int = {b:a for a, b in dict_int2aa.items()}
    dict_aa2int.update({'-':20})

# Calculate minimum Hamming discance of a sequence from MSA.

# Use this function only when computing plmDCA energy
def aa2int(seq,dict_aa2int):
    '''
    translate seq to seq_int
    
    **Arguments**
    - seq = sequence string of aa residue one-letter codes
    - dict_aa2int = dictionary of aa one-letter codes to integers
    '''
    seq_int = []
    for i in list(seq):
        seq_int.append(dict_aa2int.get(i))
    seq_int = np.array(seq_int)
    
    return seq_int

def int2aa(seq_int,dict_int2aa):
    '''
    translate seq_int to seq
    
    **Arguments**
    - seq_int = sequence string of integers as numpy vector
    - dict_int2aa = dictionary of integers to aa one-letter codes
    '''
    seq = ''.join(dict_int2aa.get(i) for i in seq_int)
    return seq

'''
@jit(nopython=True)
def computeP(seq_int,h,J,M):

    compute the probability associated with a particular integer sequence seq_int
    
    **arguments**
    - seq_int = sequence string of integers as numpy vector
    - h[ri,i] = KxM vector of external fields
    - J[ri,rj,i,j] = KxKxMxM matrix of pairwise couplings
    - M = no. positions in sequence

    assert len(seq_int) == M
    
    P=0
    for i in range(M):
        P+=h[seq_int[i],i]
        for j in range(i+1,M):
            P+=J[seq_int[i],seq_int[j],i,j]
    return P #exp(-P)

def record(records, H, J, M):
    records_toreturn = []
    for n in range(len(records)):
        # error checking
        assert len(records[n]) == M

        # translating aa -> int
        seq_int = aa2int(records[n], plm_dict.dict_aa2int)

        # computing probability from DCA Hamiltonian
        P=computeP(seq_int,H,J,M)
        records_toreturn.append(P)
    return np.array(records_toreturn)
'''
def minHamming(seq,ref_list):
    return min(np.count_nonzero(ref_list - seq,axis = 1))

def Hamming_list(seq_list,ref_list):
    return [minHamming(i, ref_list) for i in seq_list] 

@jit(nopython=True)
def sample_seq(seed, q, n, q_n, i, v_gen, method = 'multinomial'):
    v_samp = np.zeros(q)
    v_samp_nothot = np.zeros(n)
    for j in range(n):
        start = np.sum(q_n[:j])
        end = np.sum(q_n[:j+1])
        
        if method == 'multinomial':
            np.random.seed(seed) # Change random seed in each iteration.
            v_samp[start:end] = np.random.multinomial(1, v_gen[i,start:end]/np.sum(v_gen[i,start:end]))
            # only throw dice once. Get the AA which was selected.
        elif method == 'argmax':
            v_samp[start:end] = np.zeros_like(v_gen[i, start:end])  # create a zero array of the same shape
            max_index = np.argmax(v_gen[i, start:end])  # find the index of the maximum value
            v_samp[start:end][max_index] = 1  # set the max index to 1          

        v_samp_nothot[j] = np.nonzero(v_samp[start:end])[0][0]
    return v_samp_nothot

# This function is currently unavailable.
def generate(model, seed, q, q_n, n, d, device, n_gen, n_samp, thresh):
    np.random.seed(seed)
    z_gen = np.random.normal(0., 1., (n_gen, d)) #generate normal distribution of random numbers
    data = torch.FloatTensor(z_gen).to(device)
    data = model.decode(data) # Use the decoding layer to generate new sequences.
    v_gen = data.cpu().detach().numpy()
    sample_list = []
    for i in range(n_gen):
        for k in range(n_samp):
            v_samp_nothot = sample_seq(seed+k, q, n, q_n, i, v_gen)
            sample_list.append(v_samp_nothot)
    return sample_list
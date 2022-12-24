#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import os
import argparse
import pickle
import multiprocessing as mp
import time
import shutil
from itertools import repeat

import toolkit
from model import *

from Bio import SeqIO
from Bio import AlignIO
    
    
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

    print('Loading data...')
    path = '../Outputs/'
    parameters = pickle.load(open(path + options.name + ".db", 'rb'))
    q_n = parameters['q_n']
    aaindex = parameters['index']
    v_traj_onehot = parameters['onehot']
    records_MSA = parameters['seq']

    N=np.size(v_traj_onehot,axis=0)
    q=np.size(v_traj_onehot,axis=1)
    n=np.size(q_n)
    
    print('Loading VAE...')
    d=3
    model = VAE(q, d, n, q_n)
    model.load_state_dict(torch.load('VAE_SH3.pyt',map_location='cpu'))
    model.eval()
        
    # Generate new sequences
    start_all = time.time()
    
    seed = options.randseed
    n_gen = options.ngen
    n_sample = options.nsamp
    
    np.random.seed(seed)
    real_nohot_list = toolkit.convert_nohot(v_traj_onehot, q_n)
    seed_list = np.random.randint(0, 2**32, 10)
    #pool = mp.Pool(mp.cpu_count())
    
    print('Start generating sequences...')
    st_time = time.time()
    
    np.random.seed(seed)
    z_gen = np.random.normal(0., 1., (n_gen, d)) #generate normal distribution of random numbers
    data = torch.FloatTensor(z_gen).to(device)
    data = model.decode(data) # Use the decoding layer to generate new sequences.
    v_gen = data.cpu().detach().numpy()
    sample_list = []
    z_list = []

    for i in range(int(n_gen/10)):
        for k in range(n_sample):
            v_samp_nothot = toolkit.sample_seq(seed+k, q, n, q_n, i, v_gen)
            sample_list.append(v_samp_nothot)
            z_list.append(z_gen[i])
            
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
            
            # write the reference sequence
            f.write(">2vkn_chainA_p001\n")
            f.write("NFIYKAKALYPYDADDAYEISFEQNEILQVSDIEGRWWKARRNGETGIIPSNYVQLIDG\n") #2vkn_chainA_p001
            
            for item in alp_new_seq[:-1]:
                f.write(">gi\n")
                f.write("%s\n" % item)
        os.system('scaProcessMSA -a output/' + filename +'.fasta -b data -s 2VKN -c A -p 0.3 0.2 0.2 0.8')
        # Note: the above line should be customes based on the protein family you chose
        os.system('scaCore -i output/' + filename +'.db')
        os.system('scaSectorID -i output/' + filename +'.db')
        
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
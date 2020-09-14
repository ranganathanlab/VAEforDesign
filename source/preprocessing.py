#!/usr/bin/env python
# coding: utf-8

# Convert to Potts & compute plmDCA energy for MSA

import toolkit
import os
import pickle
import argparse
import numpy as np

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("alignment", help='Input Sequence Alignment')
    parser.add_argument("-n", dest ="name", default='protein', type=str, help="Name of your protein.")
    parser.add_argument("-i", dest ="idin", action="store_true", default=False, help="Identical index for all positions")
    options = parser.parse_args()
    
    savepath='../Outputs/'
    
    sequence = toolkit.get_seq(options.alignment)
    
    print("Number of sequences:",len(sequence))
    print("Number of positions:",len(sequence[0]))
    
    print('\nStart converting MSA sequences to one-hot representation...')
    aa_at_pos = toolkit.potts_index(sequence)
    if options.idin: # minimize number of features for small sample size to avoid overfitting
        aa, q_n = toolkit.convert_potts(sequence, ['-ACDEFGHIKLMNPQRSTVWY']*len(sequence[0]))
    else:
        aa, q_n = toolkit.convert_potts(sequence, aa_at_pos)
    print('q =',sum(q_n))

    D = {}
    D['index'] = aa_at_pos
    D['q_n'] = np.array(q_n)
    D['seq'] = [str(i) for i in sequence]
    D['onehot'] = aa
    pickle.dump(D, open(savepath + options.name + ".db", "wb"))
    print('Saved.')

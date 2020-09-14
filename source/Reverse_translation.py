#!/usr/bin/env python
# coding: utf-8

# http://www.kazusa.or.jp/codon/cgi-bin/showcodon.cgi?species=4932&aa=1&style=N

# In[3]:

import sys
import numpy as np
import argparse
from numba import jit
import pandas as pd
from Bio.Seq import Seq

import xlwt 
from xlwt import Workbook 
import toolkit

parser = argparse.ArgumentParser()
parser.add_argument("alignment", help='Input Sequence Alignment')
options = parser.parse_args()

savepath='../Outputs/'

CodonUsageTable = pd.ExcelFile('yeast_codon.xlsx').parse().set_index('aa')
new1500b, newheaders = toolkit.get_seq(options.alignment, get_header = True)
new1500b = [str(i) for i in new1500b]
# primers
fw_primer = 'CCGGTTGTACCTATCGAGTG'  
rv_primer = 'GTACCTCTCCTTGCATGGTC'  

# restrict enzyme (RE) sites
bamh1 = 'GGATCC'  
ecor1 = 'GAATTC'  

N = len(new1500b) # number of new sequences
n = len(new1500b[0]) # sequence length


# In[3]:


# define functions to use.
def sampling(aa, transdict, randstate):
    # Sample the codon for a single amino acid position (return nothing if the position is gap)
    if aa == '-':
        return ''
    elif np.size(TransDict[aa]['frequency']) ==1: 
        #For AA with only one codon
        return(TransDict[aa]['codon'])
    else: 
        # for AA with more than one codons
        np.random.seed(randstate)
        sample_tmp = np.random.multinomial(1,np.array(TransDict[aa]['frequency']))
        sample_index = np.where(sample_tmp!=0)[0]
        return TransDict[aa]['codon'][sample_index][0]

@jit(nopython=True)
def lcs(X, Y): 
    # https://www.geeksforgeeks.org/python-program-for-longest-common-subsequence/
    # compute longest common subsequence of two gene fragments
    
    # find the length of the strings 
    m = len(X) 
    n = len(Y) 
  
    # declaring the array for storing the dp values 
    L = [[0]*(n + 1) for i in range(m + 1)] 
  
    """Following steps build L[m + 1][n + 1] in bottom up fashion 
    Note: L[i][j] contains length of LCS of X[0..i-1] 
    and Y[0..j-1]"""
    for i in range(m + 1): 
        for j in range(n + 1): 
            if i == 0 or j == 0 : 
                L[i][j] = 0
            elif X[i-1] == Y[j-1]: 
                L[i][j] = L[i-1][j-1]+1
            else: 
                L[i][j] = max(L[i-1][j], L[i][j-1]) 
  
    # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1] 
    return L[m][n] 

def compare_lcs(codon_list, primer):
    # compute longest common subsequence (lsc) of the primer with every length = 21 substring of 
    # the gene,  and make sure no lsc > 18.
    for i in codon_list:
        compare = i[:21]
        for j in range(1, len(i)-21):
            if lcs(compare, primer) >18:
                print(i)
                break
            else:
                compare = i[j:j+21]


# In[4]:


TransDict = {} # dictionary to map amino acid with codons. One AA corresponds to >=1 codons
for i in set(CodonUsageTable.index): # Add dictionary for each amino acid
    TransDict.update({i:CodonUsageTable.loc[i]})


# In[5]:


print('Start reverse translation...')
protein = new1500b
np.random.seed(0)
seedlist = np.random.randint(50000, size = [N,n])
codon_list = []
problematic_seq = []

for i, seq in enumerate(protein):
    codon = ''
    for j, aa in enumerate(seq):
        codon += sampling(aa, TransDict, seedlist[i,j]) # sampling codon for each position
    codon = codon.replace('U','T')
    
    iter_thresh = 0
    while(bamh1 in codon or ecor1 in codon): # may need multiple trial to avoid RE sites in the oligo
        codon = ''
        for j, aa in enumerate(seq):
            codon += sampling(aa, TransDict, seedlist[i,j]+iter_thresh) 
            # change random number for another reverse translation trial
        codon = codon.replace('U','T')
        iter_thresh +=1
        
        if iter_thresh>20:
            print(seq)
            problematic_seq.append(seq) # problematic seqs can't avoid RE sites even after many trials...
            break
    if iter_thresh<=20:
        codon_list.append(codon)
        
    if i>0 and i%500 ==0:
        print('%d seqs finished...' %i)


# In[6]:


print('Verify reverse translation...')
for i, seq in enumerate(codon_list):
    assert Seq(seq).translate() == new1500b[i].replace('-',''), 'Wrong codon'
    assert bamh1 not in seq and ecor1 not in seq, 'RE sites in seq'
    assert 'U' not in seq, 'Uracil'


# In[7]:


print('Check if all oligos don’t contain any fragment with Levenshtein distance < 2 from the primers...')
print_list = ['fw_primer', 'rv_primer', 'fw_primer_rc', 'rv_primer_rc']
piter = 0
for primer in [fw_primer, rv_primer, 
               str(Seq(fw_primer).reverse_complement()), str(Seq(rv_primer).reverse_complement())]:
    compare_lcs(np.array(codon_list), primer)
    
    print('%s finished.' % print_list[piter])
    piter += 1


# In[8]:


print('Add primers and RE sites to oligos...')
CodonREPrimer_list = []
for i in codon_list:
    CodonREPrimer_list.append(fw_primer+bamh1+i+ecor1+rv_primer)


# In[9]:


# Compute length of random sequence to make total length = 300 before adding random sequences.
len_randseq = []
for i in CodonREPrimer_list:
    len_randseq.append(300 - len(i))


# In[10]:


# make sure every random sequence doesn’t contain any fragment with Levenshtein distance<2 from RE sites.
print('Generating random sequences...')
randseq_list = []
for num, i in enumerate(len_randseq):
    np.random.seed(seedlist[num,0])
    randseq = ''.join(['ACTG'[i] for i in np.random.randint(0,4,i)])
    
    niter = 1
    while('GGAT' in randseq or 'GAATT' in randseq): # GGAT + fw_primer = bamh1
        np.random.seed(seedlist[num,0] + niter)
        randseq = ''.join(['ACTG'[i] for i in np.random.randint(0,4,i)])
        niter += 1
    randseq_list.append(randseq)


# In[11]:


print('Check if all random seqs don’t contain any fragment with Levenshtein distance < 2 from the primers...')
piter = 0
for primer in [fw_primer, rv_primer, 
               str(Seq(fw_primer).reverse_complement()), str(Seq(rv_primer).reverse_complement())]:
    compare_lcs(np.array(randseq_list), primer)
    print('%s finished.' % print_list[piter])
    piter += 1


# In[12]:


print('Append random seqs to the rest part to make total length = 300...')
final300 = []
for i, seq in enumerate(CodonREPrimer_list):
    final300.append(randseq_list[i] + seq)


# In[13]:


print('Writing fasta file...')
with open(savepath + 'oligo_final.fasta', 'w') as f:
    for i,item in enumerate(final300):
        f.write(">%s\n" % newheaders[i])
        f.write("%s\n" % item)


# In[4]:


print('Writing excel file...')
wb = Workbook() 
sheet1 = wb.add_sheet('Sheet 1') 
for i,item in enumerate(final300):
    sheet1.write(i, 0, newheaders[i]) 
    sheet1.write(i, 1, item) 
wb.save(savepath + 'oligo_final.xls') 


# from Bio import SeqIO
# NatureCodon = SeqIO.parse('../twist_red_seqs_12.11.19.an', 'fasta')
# NatureCodon = [i.seq for i in NatureCodon]
# for i in NatureCodon:
#     print(i)

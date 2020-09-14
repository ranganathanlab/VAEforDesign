# Protein design VAE
---
> 09.2019
>
> Copyright (C) 2019 Xinran Lian, Andrew Ferguson, Rama Ranganathan
>
  
Protein design variational autoencoder (VAE) is an approach for designing new proteins from primary sequential structure and evolutionary constrains based on deep learning. Feeded by neural network with natural sequences (namely multiple sequence alignment, MSA), the VAE encodes the high-dimensional sequence data into low-dimensional latent space, then decodes the sample points from the latent space to construct new sequences.   
  
This repository includes a complete pipeline from preprocessing the MSA data to pick up new VAE generated sequences. The tutorial file *VAE_SH3.ipynb* is distributed as Jupyter notebook; for details please see: https://jupyter.org/. 

|            |                                                         |
| :---       | :---                                                    |
| Inputs/      | Input data, including the MSA (*.fasta*) and the SCA reference (*.pdb*).    |
| Outputs/    | Output files  |
| sources/      | The VAE codes (explained in the appendix)                |
| pySCA/     | The [pySCA module](https://github.com/ranganathanlab/pySCA)|
| VAE_SH3.ipynb | The VAE tutorial for SH3         |
| runsca_SH3.sh | The shell script to run SCA for MSA |

Getting Started
---
## 1. Clone modules from git
To clone this VAE repository together with the SCA submodule, go to your destination folder and run following commands in the terminal:  
  
```shell
git clone https://github.com/andrewlferguson/protein-design-VAE.git  
cd protein-design-VAE/SH3_v1_1  
git clone https://github.com/ranganathanlab/pySCA.git  
```  

## 2. Set up environment and dependencies
We recommand you to set up the working environment in the following steps:  
* Installed the newest version of required python packages: **numpy, pandas, numba, scipy, matplotlib, torch, sklearn, Bio**:  
  I. It is recommended to install torch according to [the official instruction](https://pytorch.org).  
  II. For the other packages, [Conda environment](https://www.anaconda.com) will be helpful. After installing conda, you can install the packages by running the command below in the terminal:    
  ```shell
  conda install numpy pandas numba scipy matplotlib sklearn Bio
  ```
  
* Install [Julia](https://julialang.org) and [plmDCA](https://github.com/pagnani/PlmDCA). plmDCA is a critical package for evaluation of the generated sequences.  
  
* Install pySCA dependencies according to [this instruction](https://ranganathanlab.gitlab.io/pySCA/install/).  
  **Note**: pySCA is a seperate module from VAE. It is highly recommended though not critical to have it installed. Without the dependencies you will not be able to add `-a` argument when generating new sequences(see below), and error will occur while running sections related to SCA in the tutorial.  

## 3. Usage
For SH3, we prepared the input *.fasta* file (*Inputs/530x129_SH3_orderedprocessed.fasta*) and reference *.pdb* structure of Sho3 (*2VKN.pdb*).  
To execute the VAE pipeline, follow the instructions at the beginning of the jupyter notebook tutorial [VAE_SH3.ipynb](VAE_SH3.ipynb).  
Briefly, run the following commands sequentially in this directory:  
```shell
cd source  
./preprocessing.py ../Inputs/530x129_SH3_orderedprocessed.fasta -n SH3  
./train_model.py -n SH3
./Generate_many_seqs.py -g 50000 -n SH3
```  
Additionally, then run the following two commands if you want to evaluate the model with SCA. 
```shell
./runsca_SH3.sh # run SCA for MSA
./Generate_many_seqs.py -n SH3 -c 1e4 -t 1.0 -p 0 -a # run SCA for generated sequences
```  
Appendix: contents of `source/` 
--- 
We have the source codes explained in three categories:
* Scripts to be execute
* Toolkits
* Scripts specified for UChicago RCC  
The SCA script for MSA is not included here because the SCA settings are **not generic** but dependent on the protein family. See the SCA documention if you are interested.
## 1. Scripts to be execute
#### *preprocessing.py*  
* Convert the MSA into one-hot Potts representation.
* Compute plmDCA probability for MSA.
```
--help
usage: train_model.py [-h] [-n NAME] [-e NBEPOCH]

optional arguments:
  -h, --help  show this help message and exit
  -n NAME     Name of your protein.
  -e NBEPOCH  number of training epochs.
```

#### *train_model.py*
* Train the VAE model.  
```
--help  
optional arguments:
  -h, --help  show this help message and exit
  -n NAME     Name of your protein.
  -e NBEPOCH  number of training epochs.
```
#### *Generate_many_seqs.py*
* Generate new sequences.
* Compute closest identity (minimum Hamming distance) of the generated sequences to MSA, plmDCA probability and the VAE log probability for generated sequences.
```
--help
usage: Generate_many_seqs.py [-h] [-g NGEN] [-s NSAMP] [-t THRESH]
                             [-p THRESHP] [-r RANDSEED] [-n NAME] [-c CUSTOM]
                             [-a]

Hint: In total ngen*nsamp new sequences are generated, default 1000. Then they
are filtered according to thresholds of plmDCA probability and minimum Hamming
distance.

optional arguments:
  -h, --help            show this help message and exit
  -g NGEN, --ngen NGEN  times of sampling in the latent space. Default 1000.
                        Recommended to enter a multiple of 10.
  -s NSAMP, --nsamp NSAMP
                        times of throwing dice at each sampling point. Default
                        10
  -t THRESH, --thresh THRESH
                        Filter out sequences with min Hamming distance larger
                        than the threshold (float). Default 1.0, meaning all 
                        sequences will be kept.
  -p THRESHP, --threshp THRESHP
                        Filter out sequences with plmDCA prob. < threshold
                        (float). Default 110.
  -r RANDSEED, --randseed RANDSEED
                        Random seed. Default 1000.
  -n NAME, --name NAME  Name of your protein.
  -c CUSTOM, --custom CUSTOM
                        A custom string for your generated sequence file name.
                        Default None.
  -a, --sca             Compute SCA for generated sequecnes
```
  
## 2. Toolkits
#### *toolkit.py*
Generic tool functions.
#### *compute_plmDCA.jl*
A julia script to compute plmDCA for MSA. Will be executed upon running *preprocessing.py*.
#### *VAE.pyt*
The trained VAE model generated upon running *train_model.py*.

## 3. Scripts specified for UChicago RCC
#### *rcc_train_model.sbatch*  
  Run *train_model.py* in UChicago RCC GPU. See [here](https://rcc.uchicago.edu/docs/using-midway/index.html) for help.
  
#### *rcc_train_model.sh*  
  Will be executed upon running *rcc_train_model.sbatch*

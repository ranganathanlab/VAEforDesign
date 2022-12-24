#!/usr/bin/env python
# coding: utf-8

# VAE examples  
# https://chrisorm.github.io/VAE-pyt.html  
# https://github.com/pytorch/examples/tree/master/vae  

import sys
import numpy as np
import os
import time
import pickle
import argparse
from model import *
import matplotlib as mpl
import matplotlib.pyplot as plt

if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
    
    
# https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/5
# grab outputs between the hidden layers

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


activation = {} 
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.cpu().detach()
    return hook

def plot_training(path, loss_train, loss_val, figname):
    
    '''
    # plotting training and validation losses over epochs of training course
    loss_train - list of training loss at each epoch of training
    loss_val - list of validation loss at each epoch of training
    ''' 
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(loss_train)), loss_train, 'b--')
    ax.plot(np.arange(len(loss_val)), loss_val, 'r--')
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.legend(["train","val"])
    if showPlots:
        plt.draw()
        plt.show()
    fig.savefig(path+'crossvalidation/'+figname, dpi=300)
    plt.close()


if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", dest ="name", default='protein', type=str, help="Name of your protein.")
    parser.add_argument("-d", dest ="dmax", default=5, type=int, help="Max dimension")
    parser.add_argument("-p", dest ="npart", default=5, type=int, help="Number of partitions")
    parser.add_argument("-e", dest ="nbepoch", default=60, type=int, help="number of training epochs.")
    options = parser.parse_args()
    
    showPlots=1
    if torch.cuda.is_available():
        print("=> Using GPU")
        print("CUDA device count =")
        print (torch.cuda.device_count())
        print("Selecting decvice = cuda:0")
        device = torch.device("cuda:0")
        print("Device name = ")
        print (torch.cuda.get_device_name(0))
    else:
        print("=> Using CPU")
        device = torch.device("cpu")

    print("Using device = %s" % device)

    # fix random seed for reproducibility
    torch.manual_seed(200186)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(200186)
    
    path = '../Outputs/'
    parameters = pickle.load(open(path + options.name + ".db", 'rb'))
    q_n = parameters['q_n']
    v_traj_onehot = parameters['onehot']
    print(v_traj_onehot.shape)
    print('number of possible amino acids in each position q_n = \n',q_n)
    print('length(q_n) = ',len(q_n))

    #define sequence based parameters
    N=np.size(v_traj_onehot,axis=0)
    q=np.size(v_traj_onehot,axis=1)
    n=np.size(q_n)

    # train/val, test split
    idx = np.arange(N)
    
    test_frac = 0.05
    val_frac = 0.20
    v_train_val, v_test, idx_train_val, idx_test = train_test_split(v_traj_onehot, idx, test_size=test_frac, random_state=np.random.randint(10000))

    print ("N = %d" % N)
    print ("N_train_val = %d" % v_train_val.shape[0])
    print ("N_test = %d" % v_test.shape[0])

    # training VAEs with d=1...dMax

    batch_size = 40
    over_batch = 5
    batches_per_epoch = np.int32(over_batch*np.ceil((1.-val_frac/(1-test_frac))*v_train_val.shape[0]/batch_size))
    
    nb_epoch = options.nbepoch
    nPartition=options.npart
    dMax=options.dmax

    start = time.time()
    argmin_val_array=np.zeros([5,5])

    train_loss_d = []
    val_loss_d = []
    train_loss_d_MSE = []
    val_loss_d_MSE = []
    
    figuredir = path+'crossvalidation'
    if os.path.isdir(figuredir)==0:
        os.mkdir(figuredir)

    for partition in range(nPartition): # k-fold cross validation

        print ('k-fold partition %d/%d' % (partition+1, nPartition))
        v_train, v_val, idx_train, idx_val, = train_test_split(v_train_val, idx_train_val, test_size=val_frac/(1-test_frac), random_state=np.random.randint(10000))

        train_loss_partiton_d = []
        val_loss_partiton_d = []

        train_loss_partiton_d_MSE = []
        val_loss_partiton_d_MSE = []

        for d in range(1,dMax+1): # iterating over different dimensionalities of latent space
            print ('k-fold partition %d/%d -- d = %d' % (partition+1, nPartition, d))

            model = VAE(q, d, n, q_n).to(device)

            loss_train = []
            loss_val = []
            loss_train_MSE = []
            loss_val_MSE = []

            for epoch in range(1, nb_epoch+1): #iterating over epochs for each dimensionality

                train_loss, val_loss, train_loss_MSE, val_loss_MSE = VAEtrain(model, epoch, batches_per_epoch, v_train, v_val)
            # append loss in every epoch    
                loss_train.append(train_loss)
                loss_val.append(val_loss)
                loss_train_MSE.append(train_loss_MSE)
                loss_val_MSE.append(val_loss_MSE)

            plot_training(path, loss_train,loss_val,'d_crossValidation_k='+str(partition+1)+'_d='+str(d)+'_loss.png')
            plot_training(path, loss_train_MSE,loss_val_MSE,'d_crossValidation_k='+str(partition+1)+'_d='+str(d)+'_lossMSE.png')

    ###############################################################################        
            # extracting min train and val loss
            argmin_val=np.argmin(loss_val)
            argmin_val_array[partition,d-1]=argmin_val

            train_loss_partiton_d.append(loss_train[argmin_val]) 
            val_loss_partiton_d.append(loss_val[argmin_val])
            train_loss_partiton_d_MSE.append(loss_train_MSE[argmin_val]) 
            val_loss_partiton_d_MSE.append(loss_val_MSE[argmin_val])
    ###############################################################################    

        # appending terminal train and val losses of current partition at all d values
        train_loss_d.append(np.asarray(train_loss_partiton_d)) 
        val_loss_d.append(np.asarray(val_loss_partiton_d))

        train_loss_d_MSE.append(np.asarray(train_loss_partiton_d_MSE)) 
        val_loss_d_MSE.append(np.asarray(val_loss_partiton_d_MSE))

    # nPartition-by-dMax arrays
    train_loss_d = np.asarray(train_loss_d) 
    val_loss_d = np.asarray(val_loss_d)

    train_loss_d_MSE = np.asarray(train_loss_d_MSE) 
    val_loss_d_MSE = np.asarray(val_loss_d_MSE)

    end = time.time()
    print("Using device = %s" % device)
    print("Elapsed time %.2f (s)" % (end - start))
    print('argmin of validation error is (rows:partition; columns: dim)',argmin_val_array)

    # averaging statisitics over cross-validation folds
    train_loss_d_mean = np.mean(train_loss_d, axis=0)
    train_loss_d_std = np.std(train_loss_d, axis=0)

    val_loss_d_mean = np.mean(val_loss_d, axis=0)
    val_loss_d_std = np.std(val_loss_d, axis=0)

    train_loss_d_mean_MSE = np.mean(train_loss_d_MSE, axis=0)
    train_loss_d_std_MSE = np.std(train_loss_d_MSE, axis=0)

    val_loss_d_mean_MSE = np.mean(val_loss_d_MSE, axis=0)
    val_loss_d_std_MSE = np.std(val_loss_d_MSE, axis=0)

    lossdic={
        'trainmean':train_loss_d_mean,
        'trainstd':train_loss_d_std,
        'valmean':val_loss_d_mean,
        'valstd':val_loss_d_std,
        'trainmean_MSE':train_loss_d_mean_MSE,
        'trainstd_MSE':train_loss_d_std_MSE,
        'valmean_MSE':val_loss_d_mean_MSE,
        'valstd_MSE':val_loss_d_std_MSE
    }
    np.save(figuredir+'loss.npy',lossdic)

    # plotting training and validation losses as function of d
    fig, ax = plt.subplots()
    plt.title('Training and validation loss as function of latent dimensionality')
    ax.plot(np.arange(dMax)+1, train_loss_d_mean, 'b--')
    ax.plot(np.arange(dMax)+1, val_loss_d_mean, 'r--')
    ax.set_xlabel("d")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.legend(["train","val"])
    plt.errorbar(np.arange(dMax)+1, train_loss_d_mean, yerr=train_loss_d_std, color='b')
    plt.errorbar(np.arange(dMax)+1, val_loss_d_mean, yerr=val_loss_d_std, color='r')
    if showPlots:
        plt.draw()
        plt.show()
    fig.savefig(figuredir+'/d_crossValidation.png', dpi=300)
    plt.close()

    # plotting MSE training and validation losses as function of d
    fig, ax = plt.subplots()
    plt.title('Training and validation loss (MSE) as function of latent dimensionality')
    ax.plot(np.arange(dMax)+1, train_loss_d_mean_MSE, 'b--')
    ax.plot(np.arange(dMax)+1, val_loss_d_mean_MSE, 'r--')
    ax.set_xlabel("d")
    ax.set_ylabel("loss (MSE)")
    ax.set_yscale("log")
    ax.legend(["train","val"])
    plt.errorbar(np.arange(dMax)+1, train_loss_d_mean_MSE, yerr=train_loss_d_std_MSE, color='b')
    plt.errorbar(np.arange(dMax)+1, val_loss_d_mean_MSE, yerr=val_loss_d_std_MSE, color='r')
    if showPlots:
        plt.draw()
        plt.show()
    fig.savefig(figuredir+'/d_crossValidation_MSE.png', dpi=300)
    plt.close()
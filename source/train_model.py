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
from model import *
import matplotlib as mpl


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


if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", dest ="name", default='protein', type=str, help="Name of your protein.")
    parser.add_argument("-e", dest ="nbepoch", default=55, type=int, help="number of training epochs.")
    options = parser.parse_args()

    if os.environ.get('DISPLAY','') == '':
        print('no display found. Using non-interactive Agg backend')
        mpl.use('Agg')

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

    # fix random seed for reproducibility
    randstate = 200
    np.random.seed(randstate)
    torch.manual_seed(randstate)
    if device == torch.device("cuda:0"):
        randstate = 2000 # RCC fela #1234
        np.random.seed(randstate)
        torch.manual_seed(randstate)
        torch.cuda.manual_seed_all(randstate)


    path = '../Outputs/'
    parameters = pickle.load(open(path + options.name + ".db", 'rb'))
    q_n = parameters['q_n']
    v_traj_onehot = parameters['onehot']
    
    print(v_traj_onehot.shape)
    print('number of possible amino acids in each position q_n = \n',q_n)
    print('length(q_n) = ',len(q_n))

    N=np.size(v_traj_onehot,axis=0)
    q=np.size(v_traj_onehot,axis=1)
    n=np.size(q_n)
    idx = np.arange(N)

    test_frac = 0.01
    val_frac = 0.20
    v_train_val, v_test, idx_train_val, idx_test = train_test_split(v_traj_onehot, idx, test_size=test_frac, random_state=randstate)

    print ("N = %d" % N)

    # ## Modification of the VAE
    # 
    # https://github.com/samsinai/VAE_protein_function/blob/master/VAE_for_protein_function_prediction.ipynb  
    # https://github.com/aniket-agarwal1999/VAE-Pytorch/blob/master/Variational%20Autoencoders.ipynb #dropout layer  
    # https://www.quora.com/What-is-the-difference-between-dropout-and-batch-normalization #batchnorm

    v_train, v_val, idx_train, idx_val, = train_test_split(v_train_val, idx_train_val, test_size=val_frac/(1-test_frac), random_state=randstate)
    print ("Training starts...")

    # training final VAE over all train_val data at optimal d
    # manually modify after finding optimal training length.
    start=time.time()

    d=3
    batch_size = 40
    over_batch = 5
    batches_per_epoch = np.int32(over_batch*np.ceil(v_train_val.shape[0]/batch_size))
    nb_epoch = options.nbepoch # Optimal nb_epoch is 55.

    model = VAE(q, d, n, q_n).to(device)

    loss_train = []
    loss_train_MSE = []
    for epoch in range(1, nb_epoch+1):
        train_loss, _, train_loss_MSE, _ = VAEtrain(model, epoch, batches_per_epoch, v_train_val, v_val)
        # training together with validation set are used together to train the final VAE.
        loss_train.append(train_loss)
        loss_train_MSE.append(train_loss_MSE)

    end = time.time()
    print("Using device = %s" % device)
    print("Elapsed time %.2f (s)" % (end - start))

    # saving trained model
    save_path = "./VAE_"+options.name+".pyt"
    torch.save(model.state_dict(), save_path)

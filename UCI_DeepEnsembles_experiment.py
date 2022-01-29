#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 17:14:49 2022

@author: bhargobdeka
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 14:27:24 2021
Deep Ensembles
@author: BD
"""
#import pandas as pd
import zipfile
import urllib.request
import os
import time
import json
import copy
import math
#import GPy
import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, TensorDataset
# from torch.optim.sgd import SGD
# from sklearn.model_selection import KFold
# from scipy.io import savemat
# from torchvision import datasets, transforms
# from torchvision.utils import make_grid
# from tqdm import tqdm, trange

#####################################
# Network Class
#####################################
# Create the net
class Net(nn.Module):
    def __init__(self, input_dim, output_dim, num_units):
        super(Net, self).__init__()
        self.input_dim  = input_dim
        self.output_dim = output_dim
        
        self.l1 = nn.Linear(input_dim, num_units)
        self.l2 = nn.Linear(num_units, 2)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x

######################################
# Create model fit function
def fit_model(model, optimizer, train_loader):
    
    for x_batch, y_batch in train_loader:
        # Reset grad and loss to zero
        x_batch = x_batch.requires_grad_(requires_grad=True)
        optimizer.zero_grad()
        
        output = model(x_batch)
        sp = torch.nn.Softplus()
        mu, sig = output[:,0], sp(output[:,1])+(10)**-6
        
        loss = nll_criterion(mu, sig, y_batch)
        # loss = loss.mean()
        ## Adverserial Learning
        loss = loss.mean()
        loss.backward(retain_graph=True)
        
        gradient = x_batch.grad.data
        gradient = torch.clamp(gradient, min=-1, max=1)
        
        x_a = x_batch + eps*(torch.sign(gradient)) ###################
        
        
        optimizer.zero_grad()
        
        output_a = model(x_a)
        mu_a, sig_a = output_a[:,0], sp(output_a[:,1])+(10)**-6
        
        loss = nll_criterion(mu, sig, y_batch) + nll_criterion(mu_a, sig_a, y_batch)
        loss = loss.mean()
        # gradient descent or adam step
        loss.backward()
        optimizer.step()
        
        
    return model

def model_predict(model_instance, X_test):
    model  = model_instance
    output = model(X_test)
    sp = torch.nn.Softplus()
    mu, sig = output[:,0], sp(output[:,1])+(10)**-6
    return mu, sig

    

###################################

# Setting the seed
# torch.manual_seed(42)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# np.random.seed(1)
# We load the data
# data_name = ["bostonHousing"]
#data_name = ["kin8nm", "naval-propulsion-plant", "power-plant", "protein-tertiary-structure"]
data_name = ["concrete","energy", "wine-quality-red", "yacht", \
              "kin8nm","naval-propulsion-plant",\
              "power-plant","protein-tertiary-structure"]
for j in range(len(data_name)):
    data = np.loadtxt(data_name[j] + '/data/data.txt')
    # We load the indexes for the features and for the target
    
    index_features = np.loadtxt(data_name[j] +'/data/index_features.txt').astype(int)
    index_target   = np.loadtxt(data_name[j] +'/data/index_target.txt').astype(int)
    
    
    # Change only for Protein
    if data_name[j] == 'protein-tertiary-structure':
        num_units = 100
        n_splits  = 5
    else:
        num_units = 50
        n_splits  = 20
        
    # Input data and output data
    X = data[ : , index_features.tolist() ]
    Y = data[ : , index_target.tolist() ]
    input_dim = X.shape[1]
    
    if os.path.isfile("results_DVI/log_{}.txt".format(data_name[j])):
        os.remove("results_DVI/log_{}.txt".format(data_name[j]))
#    from subprocess import call
#    call(["rm", "results/log_{}.txt".format(data_name[j])], shell=True)
    _RESULTS_lltest = data_name[j] + "/results_DE/lltest_DVI.txt"
    _RESULTS_RMSEtest = data_name[j] + "/results_DE/RMSEtest_DVI.txt"
    
    nll_criterion = lambda mu, sigma, y: torch.log(sigma)/2 + ((y-mu)**2)/(2*sigma + 1e-06)
    # torch.log(sigma) + 0.5*np.log(2*np.pi)
    # Eval functions used to compute final loss
    nll_eval = lambda mu, sigma, y: -0.9189 - np.log(sigma)/2 - ((y-mu)**2)/(2*sigma + 1e-06)
    mse_eval = lambda mu, y: np.sqrt(np.mean(np.power(mu-y,2)))
        
    train_LLs, test_LLs, train_RMSEs, test_RMSEs, runtimes = [], [], [], [], []
    eps = 0.01
    
    runtimes = []
    lltests, rmsetests = [], []
    for i in range(n_splits):
        index_train = np.loadtxt(data_name[j] +"/data/index_train_{}.txt".format(i)).astype(int)
        index_test = np.loadtxt(data_name[j] +"/data/index_test_{}.txt".format(i)).astype(int)
        
        #Check for intersection of elements
        ind = np.intersect1d(index_train,index_test)
        if len(ind)!=0:
            print('Train and test indices are not unique')
            break
        
        # Train and Test data for the current split
        X_train = X[ index_train.tolist(), ]
        Y_train = Y[ index_train.tolist() ]
        Y_train = np.reshape(Y_train,[len(Y_train)]) #BD
        X_test  = X[ index_test.tolist(), ]
        Y_test  = Y[ index_test.tolist() ]
        Y_test = np.reshape(Y_test,[len(Y_test)])    #BD
        
        # Normalise Data
        X_means, X_stds = X_train.mean(axis = 0), X_train.var(axis = 0)**0.5
        idx = X_stds==0
        X_stds[np.where(idx)[0]] = 1
        Y_means, Y_stds = Y_train.mean(axis = 0), Y_train.var(axis = 0)**0.5
    
        X_train = (X_train - X_means)/X_stds
        Y_train = (Y_train - Y_means)/Y_stds
        # X_val   = (X_val - X_means)/X_stds
        # Y_val   = (Y_val - Y_means)/Y_stds
        X_test  = (X_test - X_means)/X_stds
        # Y_test  = (Y_test - Y_means)/Y_stds
        # Converting to Torch objects
        X_train = torch.from_numpy(X_train).float()
        Y_train = torch.from_numpy(Y_train).float()
        # X_train = torch.Tensor(X_train, dtype = torch.float, requires_grad=True)
        # Y_train = torch.Tensor(X_train, dtype = torch.float, requires_grad=True)
        # X_val   = torch.from_numpy(X_val).float()
        # Y_val   = torch.from_numpy(Y_val).float()
        
        X_test = torch.from_numpy(X_test).float()
        Y_test = torch.from_numpy(Y_test).float()
        
        # Creating tensor Dataset
        train_data = TensorDataset(X_train, Y_train)
        
        # Creating test tensor Dataset
        test_data = TensorDataset(X_test, Y_test)
        
        # initialize 5 networks
        n = 5
        models = []
        for i in range(n):
            model=Net(input_dim, output_dim=1, num_units=num_units)
            op = torch.optim.Adam(model.parameters(), lr = 0.01)
            models.append((model,op))
        # print(models)
        
        
        # Train Loader
        train_loader = DataLoader(dataset=train_data, batch_size=128, shuffle=True)
        
        # Test Loader
        test_loader = DataLoader(dataset=test_data, batch_size=500)
        
        train_logliks, test_logliks, train_errors, test_errors = [], [], [], []
        val_logliks = []
        
        # start_time
        start_time = time.time()
        num_epochs = 400
        
        
        
        mzstack, Szstack = [],[]
        errors, lls = [],[]
        start_time = time.time()
        for epoch in range(num_epochs):
            
            for model, op in models:
                model = fit_model(model, op, train_loader)
                mz, Sz = model_predict(model, X_test)
                mzstack.append(mz)
                Szstack.append(Sz)
                # error, ll = evaluate_n_models(models, i, X_test, Y_test)
                # errors.append(error)
                # LLs.append(ll)
            mZstack = torch.stack(mzstack).detach().numpy()
            SZstack = torch.stack(Szstack).detach().numpy()
            
            mz_ensemble = np.mean(mZstack,axis=0)
            # Sz_ensemble = np.mean(np.power(mZstack, 2) + SZstack, axis = 0) - np.power(mz_ensemble, 2)
            Sz_ensemble = np.mean(SZstack + np.power((mZstack- mz_ensemble),2),axis = 0)
            # Test RMSE and LL
            ## Metrics
            Y_true = Y_test.detach().numpy()
            test_ystd  = np.std(Y_true)
            test_ymean = np.mean(Y_true)
            
            y_mean = test_ystd*mz_ensemble + test_ymean
            y_std  = (test_ystd**2) * Sz_ensemble
            
            
            rmse    = np.sqrt(np.mean(np.power((Y_true - y_mean),2)))
            test_ll = np.mean(-0.5 * np.log(2 * np.pi * (y_std)) - \
                          0.5 * (Y_true - y_mean)**2 / (y_std))
            
            # print("RMSE : "+ str(rmse))
            # print("Test LL : "+ str(test_ll))
            errors += [rmse]
            lls += [test_ll]
        runtime = time.time()-start_time
        lltests.append(lls)
        rmsetests.append(errors)
        runtimes.append(runtime)
    
    mean_ll   = np.mean(lltests,axis=0)
    mean_RMSE = np.mean(rmsetests,axis=0)
    print("best LL"+str(mean_ll[-1]))
    
    plt.scatter(range(400), mean_ll)
    plt.show()
    plt.scatter(range(400), mean_RMSE)
    plt.show()
      
    with open(_RESULTS_lltest, "w") as myfile:
           for item in mean_ll:
                   myfile.write('%f\n' % item)
    with open(_RESULTS_RMSEtest, "w") as myfile:
           for item in mean_RMSE:
               myfile.write('%f\n' % item)
       
    with open("results_DE/log_{}.txt".format(data_name[j]), "a") as myfile:
               myfile.write('Avg. runtime is %f +- %f  \n' % (np.mean(runtimes), np.std(runtimes)))
                     
            
            
            
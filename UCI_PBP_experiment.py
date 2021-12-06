# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 22:11:48 2021

@author: BD
"""

import math
import os
import numpy as np
import time
import sys
import matplotlib.pyplot as plt
multiplier = [10]
for mul in range(len(multiplier)):
    data_name = ["protein-tertiary-structure"]
#    data_name = ["concrete","energy","wine-quality-red",\
#                  "yacht","kin8nm", "naval-propulsion-plant",\
#                  "power-plant", "protein-tertiary-structure"]
    for j in range(len(data_name)):
        sys.path.append(data_name[j] +'/PBP_net/')
        import PBP_net
        
        # We delete previous results
        # Delete log files
        if os.path.isfile("results_{}xepoch_PBP/log_{}.txt".format(multiplier[mul], data_name[j])):
            os.remove("results_{}xepoch_PBP/log_{}.txt".format(multiplier[mul], data_name[j]))
    #    from subprocess import call
    #    call(["rm", "results/log_{}.txt".format(data_name[j])], shell=True)
        _RESULTS_lltest = data_name[j] + "/results_PBP/lltest_PBP.txt"
        _RESULTS_RMSEtest = data_name[j] + "/results_PBP/RMSEtest_PBP.txt"
        # We fix the random seed
        seed_list = [1]
        for s in seed_list:
            np.random.seed(s)
            
            # We load the data
            
            data = np.loadtxt(data_name[j] +'/data/data.txt')
            
            # We load the number of hidden units
            
            n_hidden = int(np.loadtxt(data_name[j] +'/data/n_hidden.txt').tolist())  #BD
            
            # We load the number of training epocs
            
            n_epochs = int(np.loadtxt(data_name[j] +'/data/n_epochs.txt').tolist())   #BD
            
            # We load the indexes for the features and for the target
            
            index_features = np.loadtxt(data_name[j] +'/data/index_features.txt')
            index_features = index_features.astype(int) # convert ndarray to int  #BD
            
            index_target = np.loadtxt(data_name[j] +'/data/index_target.txt')
            index_target = index_target.astype(int)    #BD
            
            X = data[ : , index_features.tolist()]
            y = data[ : , index_target.tolist() ]
            
            # We iterate over the training test splits
            
            n_splits = int(np.loadtxt(data_name[j] +'/data/n_splits.txt'))  #BD
            
            errors, lls, times = [], [], []
            lltests, RMSEtests = [], []
            for i in range(n_splits):
            
                # We load the indexes of the training and test sets
            
                index_train = np.loadtxt(data_name[j] + "/data/index_train_{}.txt".format(i)).astype(int)
                index_test = np.loadtxt(data_name[j] + "/data/index_test_{}.txt".format(i)).astype(int)
            
                X_train = X[ index_train.tolist(), ]
                y_train = y[ index_train.tolist() ]
                X_test = X[ index_test.tolist(), ]
                y_test = y[ index_test.tolist() ]
                
                # We construct the network
            
                # We iterate the method 
                start_time = time.time()
                network = PBP_net.PBP_net(X_train, y_train, X_test, y_test,
                    [ n_hidden], normalize = True, n_epochs = n_epochs*multiplier[mul], testing = True)
                running_time = time.time() - start_time
                lltest   = network.lltests
                RMSE     = network.RMSE
                # We obtain the test RMSE and the test ll
            
                # We make predictions for the test set

                m, v, v_noise = network.predict(X_test)
                
                # We compute the test RMSE
                
                rmse = np.sqrt(np.mean((y_test - m)**2))
                test_ll = np.mean(-0.5 * np.log(2 * math.pi * (v + v_noise)) - \
                                  0.5 * (y_test - m)**2 / (v + v_noise))
                print("RMSE : "+ str(rmse))
                print("Test LL : "+ str(test_ll))
                errors += [rmse]
                lls += [test_ll]
                times += [running_time]
                lltests.append(lltest)
                RMSEtests.append(RMSE)
                
            print("Avg. test LL is %f +- %f" % (np.mean(lls), np.std(lls)))
            print("Avg. test RMSE is %f +- %f" % (np.mean(errors), np.std(errors)))
            print("Avg. time is %f +- %f" % (np.mean(times), np.std(times)))
            
            mean_ll = np.mean(lltests,axis=0)
            mean_RMSE = np.mean(RMSEtests,axis=0)
            
            with open(_RESULTS_lltest, "w") as myfile:
                for item in mean_ll:
                    myfile.write('%f\n' % item)
            with open(_RESULTS_RMSEtest, "w") as myfile:
                for item in mean_RMSE:
                    myfile.write('%f\n' % item)
        
            plt.scatter(range(400), mean_ll)
            plt.show()
            plt.scatter(range(400), mean_RMSE)
            plt.show()
            
            with open("results_{}xepoch_PBP/log_{}.txt".format(multiplier[mul], data_name[j]), "a") as myfile:
                myfile.write('random seed %d \n' % (s))
                myfile.write('Avg. train LL is %f +- %f  \n' % (np.mean(lls), np.std(lls)))
                myfile.write('Avg. test RMSE is %f +- %f  \n' % (np.mean(errors), np.std(errors)))
                myfile.write('Avg. runtime is %f +- %f  \n' % (np.mean(times), np.std(times)))
               
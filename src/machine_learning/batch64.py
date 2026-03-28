#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 13:37:49 2020

@author: duarte
"""

SEED = 127
import time
start_time = time.time()
import os
import random
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd


from random import sample
from sklearn import preprocessing


def NORMALIZE(array,mean,std):
    return (array - mean)/std
def UNNORMALIZE(normalized_array,mean,std):
    return normalized_array * std + mean


random.seed(SEED)

data_dir = os.path.join('/home/duarte/Documents/03 - q or no q', "dataset.csv") #< -----------
data = pd.read_csv(data_dir)
df =  pd.DataFrame(data)
df =  df.dropna()



X = df.loc[ : , ["M_chirp","q","Lambda_tilda"]]
Y = df.loc[ : , "R1":"Lambda2"]

## Divide data into 75% train, 25% test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,
                                                   test_size=0.25,  #<------
                                                   random_state=1)
########################################################
#################NORMALIZE DATA#########################
########################################################
mean_X_train = X_train.mean()
mean_Y_train = Y_train.mean()

sd_X_train = X_train.std()
sd_Y_train = Y_train.std()

X_train_scaled = (X_train - mean_X_train)/sd_X_train
Y_train_scaled = (Y_train - mean_Y_train)/sd_Y_train
X_test_scaled  = (X_test  - mean_X_train)/sd_X_train
Y_test_scaled  = (Y_test  - mean_Y_train)/sd_Y_train
###############SAVE NORMAL. CONSTANTS#################

mean_df = pd.DataFrame(mean_X_train)
std_df  = pd.DataFrame(sd_X_train)

mean_df.to_csv('mean.csv')
std_df.to_csv('std.csv')

####################SPLIT DATA########################
######################INTO############################
#################TRAIN/VALIDATION#####################
X_train_sample, X_val, Y_train_sample, Y_val = train_test_split(X_train_scaled, Y_train_scaled,
                                                  test_size=0.2,random_state=1)

X_val = np.array(X_val)
Y_val = np.array(Y_val)
####################SHUFFLE###########################
X_train = np.array(X_train_sample)
Y_train = np.array(Y_train_sample)
#
X_test = np.array(X_test_scaled)
Y_test = np.array(Y_test_scaled)

ind = np.random.permutation(len(X_train))
ind2 = np.random.permutation(len(X_test))

X_train = np.take(X_train, ind, axis=0)
Y_train = np.take(Y_train, ind, axis=0)

X_test  = np.take(X_test, ind2, axis=0)
Y_test  = np.take(Y_test, ind2, axis=0)


####################IMPORTS###########################
from keras.models import Sequential,model_from_json
from keras import optimizers
from keras.layers import Dense, Dropout #, Conv1D, MaxPool1D, Flatten, SpatialDropout1D
from sklearn.preprocessing import LabelEncoder
#from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

from IPython.display import clear_output
######################################################
A4='relu'
A2='sigmoid'
A3='tanh'
A1='softmax'
lista=[A4]
#lista = [A1,A2,A3,A4]
for Ai in lista:
    
    ########################################################
    #################MODEL STRUCTURE########################
    ########################################################
    model = Sequential()
    model.add(Dense(16,activation=Ai, input_shape=[X_train.shape[1]]))
    model.add(Dropout(0))
    model.add(Dense(64,activation=Ai))
#    model.add(Dropout(0))
    model.add(Dense(32, activation=A3))
    model.add(Dense(6))
    
    ########################################################
    ########################################################
    opt = optimizers.Adam(lr=0.001)
    
    model.compile(opt, loss ="mse", metrics =["accuracy"])
    
    epocas = 1000                                   #<------------------------
    history = model.fit(X_train, Y_train,
              epochs=epocas, batch_size=64,
              validation_data= (X_val,Y_val))
    

    ########################################################
    ########################################################
    
    loss_train = history.history['loss']
    loss_val = history.history['val_loss']#*sd_Y_train**2
    print(loss_train)
    
    ########################################################
    ################SAVE MODEL and WEIGHTS##################
    ########################################################
    #Here I'm saving the model to a .json file (?) and the #
    ##########weights of the model on a .h5 file############
    json_model_name = str(Ai) + "_model50q1.json"
    hdf5_model_name = str(Ai) + "_model50q1.h5"
    

    s= "mse:" + str(round(loss_train[-1],5))
    
    model_json = model.to_json()
    with open(json_model_name, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(hdf5_model_name)
    print("Saved ", json_model_name, " to disk")
    
    ########################################################
    ##################PLOT AND SAVE JPG#####################
    ########################################################    
    
    plt.plot(loss_train)
    plt.plot(loss_val)
    plt.yscale('log')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    file_name = str(epocas) + " epoch5 "+ str(Ai) + ".jpg" 
    plt.savefig(file_name)
    plt.show()
    
    plt.plot(loss_train)
    plt.plot(loss_val)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    file_name = str(epocas) + " epoch5 "+ str(Ai) + ".jpg" 
    plt.savefig(file_name)
    plt.show()
    
    
    plt.plot(history.history['accuracy'])
    plt.show()
    
    #########################################################
    
    Y_predicted = np.array(model.predict(X_test))
    
    
    mean_Y_train = np.array(mean_Y_train)
    sd_Y_train   = np.array(  sd_Y_train)
    
    y_predicted = UNNORMALIZE(Y_predicted, mean_Y_train, sd_Y_train )
    Y_test      = UNNORMALIZE(Y_test     , mean_Y_train, sd_Y_train )
    diferença = (y_predicted - Y_test)**2
    diferença = diferença.T
    
    MSE = []
    
    for i in range (6):
        MSE.append(np.mean(diferença[i]))
        
    MD = np.sqrt(MSE)
    
    print(MD)
    
#    UNNORMALIZE(MSE, mean_Y_train*0, sd_Y_train )
        
    
    
    
    
time_elapsed = time.time()-start_time

print ('Time Elapsed: ', time_elapsed)
#history_dict.keys()

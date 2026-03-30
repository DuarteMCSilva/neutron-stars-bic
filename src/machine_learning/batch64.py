#/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 18:23:33 2019

@author: duarte
"""
SEED = 123
import time
start_time = time.time()
import os
import random
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd


import utils.normalization as norm


random.seed(SEED)

data_dir = os.path.join('./data/binary-systems.csv')
data = pd.read_csv(data_dir)
df =  pd.DataFrame(data)



X = df.loc[ : ,"K_sat":"Q_sym"]
Y = df.loc[ : ,"Lambda14"]

## Divide data into 75% train, 25% test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test= train_test_split(X,Y,
                                                   test_size=0.25,
                                                   random_state=1)

########################################################
#################NORMALIZE DATA#########################
########################################################

mean_df = pd.DataFrame(X_train.mean())
std_df  = pd.DataFrame(X_train.std())

mean_df.to_csv(OUTPUT_PATH + 'mean.csv')
std_df.to_csv(OUTPUT_PATH + 'std.csv')

sd_Y_train = Y_train.std()

X_train_scaled = norm.normalize(X_train)
Y_train_scaled = norm.normalize(Y_train)
X_test_scaled  = norm.normalizeWithReference(X_test, X_train)
Y_test_scaled  = norm.normalizeWithReference(Y_test, Y_train)

####################SPLIT DATA########################
######################INTO############################
#################TRAIN/VALIDATION#####################
X_train, X_val, Y_train, Y_val = train_test_split(X_train_scaled, Y_train_scaled,
                                                  test_size=0.2,random_state=1)
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
    model.add(Dense(32,activation=Ai, input_shape=[X_train.shape[1]]))
    model.add(Dropout(0.0))
    model.add(Dense(32,activation=Ai))
    model.add(Dropout(0.0))
    model.add(Dense(64, activation=Ai))
    model.add(Dense(1))
    
    ########################################################
    ########################################################
    opt = optimizers.Adam(lr=0.001)
    
    model.compile(opt, loss ="mse", metrics =["accuracy"])
    
    epocas = 5
    history = model.fit(X_train, Y_train,
              epochs=epocas, batch_size=64,
              validation_data= (X_val,Y_val))
    
    y_predicted = np.array(model.predict(X_test))
    ########################################################
    ########################################################
    
    loss_train = history.history['loss']
    loss_val = history.history['val_loss']#*sd_Y_train**2
    
    ########################################################
    ################SAVE MODEL and WEIGHTS##################
    ########################################################
    #Here I'm saving the model to a .json file (?) and the #
    ##########weights of the model on a .h5 file############
    json_model_name = str(Ai) + "_model1.json"
    hdf5_model_name = str(Ai) + "_model1.h5"
    

    s= "mse:" + str(round(loss_train[-1],5))
    
    model_json = model.to_json()
    with open(json_model_name, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(hdf5_model_name)
    print("Saved ", json_model_name, " to disk")
    
    ########################################################
    ##################PLOT AND SAVE JPG#####################
    ########################################################
    
    plt.plot(np.array(loss_train)*sd_Y_train)
    plt.plot(np.array(loss_val)*sd_Y_train)
    plt.text(epocas*0.75-1, round(loss_train[0] * 0.8,4) , s )
    plt.text(epocas*0.75-1, round(loss_train[0] * 0.5,4) , 'lr = 0.001')
    plt.text(epocas*0.75-1, round(loss_train[0] * 0.3,4) , Ai)
    plt.ylabel('loss function (mse) ')
    #plt.plot(history.history['val_loss'])
    #plt.title('model loss')
    #plt.ylabel('loss')
    plt.yscale('log')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    
    file_name = str(epocas) + " epoch "+ str(Ai) + "0.jpg" 
    plt.savefig(file_name)
    plt.show()
    #########################################################

time_elapsed = time.time()-start_time

print ('Time Elapsed: ', time_elapsed)

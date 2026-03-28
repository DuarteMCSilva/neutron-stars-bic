#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 13:19:48 2020

@author: duarte
"""


import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd
#import keras

from matplotlib.ticker import MaxNLocator
#
from matplotlib.colors import BoundaryNorm


def NORMALIZE(array,mean,std):
    return (array - mean)/std
def UNNORMALIZE(normalized_array,mean,std):
    return normalized_array*std+mean


X_mean = [1.191744177, 631.62232275]
X_std  =[0.11530894, 359.68684556]

Title = ['R1','M1','L1','R2','M2','L2']
Y_mean = [12.2019949, 1.31071097, 966.138343, 12.16374989, 1.474409255, 867.6562185]
Y_std  = [0.21058501, 0.17228159, 758.227512, 0.286395225, 0.334088814,1188.4079897]


file = "Opt2"

M_chirp_mean  = 1.1917441775
Lambda_T_mean = 631.62232275
q_mean        = 1.1633465093

M_chirp_std   = 0.1153089432
Lambda_T_std  = 359.68684556
q_std         = 0.3732815414
###################INPUT#########################

dot = int(200)
M_chirp  = np.full(dot**2 , 1.186)
q = np.arange(0.70, 1.0, 0.3/dot).T
Lambda_T = np.arange(0. , 730., 730./dot)


M_chirp  = M_chirp[:dot**2]
q        = q[:dot]
Lambda_T = Lambda_T[:dot]


#################NORMALIZATION###################
M_chirp  = NORMALIZE(M_chirp,M_chirp_mean,M_chirp_std)
q        = NORMALIZE(q       , q_mean      , q_std)
Lambda_T = NORMALIZE(Lambda_T,Lambda_T_mean,Lambda_T_std)
#################################################

q, Lambda_T = np.meshgrid( q, Lambda_T)  
########################################################################


x  = M_chirp.reshape(dot**2,)
y  = q.reshape(dot**2,)
w  = Lambda_T.reshape(dot**2,)

adata = np.array([x,y,w]).T
from keras.models import model_from_json #,Sequential
'''-------------Abrir o modelo, ler, tratar, prever-----------------'''

Name = file  ###<------ Here you can change the model.
Model_name = Name + '.json' 
Weights    = Name + '.h5'

json_file = open(Model_name , 'r')
Model_json = json_file.read()
json_file.close()

Model = model_from_json(Model_json)
Model.load_weights(Weights)
print("Loaded model from disk: ", Model)
#R1,M1,L1,R2,M2,L2= np.array(Model.predict(adata)).T
PREDICTS = np.array(Model.predict(adata)).T
#PREDICTS = PREDICTS.astype('float64')
'''------------------------------------------------------------------'''

'''------------------------------------------------------------------'''

Lambda_T = UNNORMALIZE(Lambda_T,Lambda_T_mean,Lambda_T_std)
q        = UNNORMALIZE(q       , q_mean      , q_std)


for i in range (len(PREDICTS)):
    z = PREDICTS[i]
    
    Z = np.array(z.reshape(Lambda_T.shape))
    Z = UNNORMALIZE(Z,Y_mean[i],Y_std[i])
    
    levels = MaxNLocator(nbins=150).tick_values(Z.min(), Z.max())
    cmap = plt.get_cmap('inferno')
    norm = BoundaryNorm(levels, ncolors = cmap.N, clip=True)
    
    plt.contourf( Lambda_T, q, Z, norm=norm, cmap = 'Blues')
    plt.colorbar()
#    CS= plt.contour(Lambda_T,M_chirp,R1, [9,10,11,12], colors='k')
#    plt.clabel(CS, inline=1)
    title = Title[i] ### <---(9/11) Change this too
    plt.title(str(title + '  M = 1.186'))
    plt.xlabel('Lambda_T')
    plt.ylabel('q')
    plt.savefig(str(Title[i] + '.jpg'))
    plt.show()
    

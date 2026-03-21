#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 18:46:03 2020

@author: Duarte Coelho Silva & Márcio Ferreira
"""

'''Index (Ctrl + F the following keyword):
        Structure:
            1- $Import
            2- $Input
            3- $Restrictions
            4- $TOVSolver
                4.1 - $TOVeq
                4.2 - $InitialCond
            5- $Execution
            6- $Organizing
            7- $Plotting
            8- $Saving
        
        Observations:
            -%OBS - Observation
            -%PAR - Parameters
            -%NOR - Normalization
    '''

#############################$Import#############################
import matplotlib.pyplot as plt 
import time

import tov.tov_solver as tov_solver

import utils.date_utils as date_utils

import os
import pandas as pd

from multiprocess import Pool # %OBS: There is a similar library called "multiprocessing", 
                                  # might be an option too but exibits some errors which "multiprocess" does not.

#############################$Input#############################
n_cores = 1 #%PAR
data_dir = os.path.join('./data/output_eos/2026-03-21_20-18-06.csv')

#Depends on the data structure
data = pd.read_csv(data_dir, sep = " ")

#############################$Restrictions#############################

df  =  pd.DataFrame(data)
#del df['Velocity of Sound']

df   = df[ df['rho'] > 0.0002]  #%PAR
df   = df[ df['rho'] <= 1.5] 
df   = df.dropna(axis = 1)
df   = df.dropna(axis = 0)

df.reset_index(drop=True, inplace=True)
indices= df['id'].unique()

df.reset_index(drop=True, inplace=True)

#############################$ToVSolver#############################

def Paralelize(DataFrame, ncores):
 
    indices = DataFrame['id'].unique()


    def EOS_array(DataFrame): # Creates a list, each element is an EoS
        EOS_list = []

        for id in indices:
            EOS = DataFrame[DataFrame['id'] == id]

            EOS_list.append(EOS)
        return EOS_list

    def Solve_for_EOS(EOS):
        solver = tov_solver.TOV_solver(N_stars = 80, rho0 = 0.05, rmin = 1.e-8, rmax=20., dr = 1.e-3)
        print(solver.rho0, solver.rmin, solver.rmax, solver.dr, solver.int_method)
        return solver.solve(EOS)
    
    EOS_array = EOS_array(DataFrame)
    Total_DataFrame = []
    
    if(ncores == 1):
        for EOS in EOS_array:
            Total_DataFrame.append(Solve_for_EOS(EOS))
    else:
        Total_DataFrame = Pool(ncores).map(Solve_for_EOS, EOS_array)   #Vetor de Dataframes
    
    return Total_DataFrame 


#############################$Execution#############################

start = time.time()
Vector_DATA_FRAME = Paralelize(df, n_cores)
end = time.time()
print(n_cores, ' cores:')
print("Time Elapsed = ",end-start ,"s")
#############################$Organizing#############################
size = len(Vector_DATA_FRAME)

VDF = []

for j in range(size):
    EoS = Vector_DATA_FRAME[j]
    
    Massas = EoS.M
    max_ind = Massas.idxmax()
    max_mass= Massas[max_ind]
    
    if max_mass>=1.97: #%PAR
        VDF.append(EoS[:max_ind]) # %OBS: Notice that any EOS that doesn't reach the max_mass won't be graphed (cannot replicate observations).
                                    #Thus, the graph might not show as many EOS's as the contained in the initial dataframe.
    
size = len(VDF)

if(size == 0):
    print('No EOS reaches the max mass of 1.97 M_sun, no graph will be plotted.')

#############################$Plotting#############################
for j in range(size):
    M = VDF[j].M
    R = VDF[j].R
    indice = VDF[j].id
    
    plt.plot(R,M, label = indice[0])
plt.title('M(R)')
#plt.legend(loc = 'upper left',  bbox_to_anchor=(1.05, 1))
plt.savefig('M(R).png')
plt.show()

for j in range(size):
    M = VDF[j].M
    L = VDF[j].Lambda
    indice = VDF[j].id
    plt.plot(M,L, label = indice[0])
plt.title('Lambda(M)')
plt.yscale('log')
#plt.legend(loc = 'upper left',  bbox_to_anchor=(1.05, 1))
plt.savefig('Lambda(M).png')
plt.show()

#############################$Saving#############################
Final_dataframe = pd.DataFrame()
Final_dataframe = pd.concat(VDF, ignore_index=True) # %OBS: This is a way of concatenating the dataframes contained in the list VDF into a single dataframe.

Create_new_file = True #%PAR
Default_name = str('R_M_L_')+ date_utils.get_current_date_string() +str('.csv')

if Create_new_file == True:
    Final_dataframe.to_csv(Default_name, index= False) ## sep=" ", 

    print("saved to the file: ", Default_name)

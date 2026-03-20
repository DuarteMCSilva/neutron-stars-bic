#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 10:55:06 2020

@author: Duarte Coelho Silva & Márcio Ferreira
"""
'''Index (Ctrl + F the following keyword):
        Structure:
            1- $Import
            2- $Input
            3- $Organization
            4- $OneEoS
                4.1 - $RandomPart
                4.2 - $AddingCrust
            5- $Execution
            6- $Saving
        
        Observations:
            -%OBS - Observation
            -%PAR - Parameters
            -%NOR - Normalization
    '''
#############################$Import#############################
import pandas as pd
import numpy as np
import time

#############################$Input#############################
N = 8  #How many EoS to be generated  (%PAR)

data = pd.read_csv('./data/crust.csv')


#############################$Organization#############################
df = pd.DataFrame(data)
df = df.loc[(df['n']<= 0.15)] #%PAR


Column_Names = list(df.columns) 
Keys = data.keys()
df = df.rename( {Keys[0]: "rho" }, axis = 1) 


#Insert a column for the Velocity of Sound
de = np.append(np.nan,np.diff(df.e))
dp = np.append(np.nan,np.diff(df.p))
cs = np.sqrt(dp/de)
df['VS'] = cs

def get_initial_conditions(df):
  last_row = df.iloc[-1,]

  print("Last Row Information:")
  print(last_row)

  n0 = last_row['rho']
  e0 = last_row['e']
  p0 = last_row['p']
  c0 = last_row['VS']

  return n0, e0, p0, c0

n0, e0, p0, c0 = get_initial_conditions(df)
#############################$OneEoS#############################

def One_EoS(crust= df, model_name= 1, quantity = 6, n_tr = 0.15, n_sat=12*0.16, min_c = 0., max_c = 1. ): #%PAR
  def Random_n():
    Extremos = np.array([n_tr,n_sat])
    A = np.random.random(quantity)

    n_points = A*(n_sat-n_tr) + n_tr
    n_points = np.append(Extremos,n_points)
    n_points = np.sort(n_points)

    return n_points

  def Random_c():
    A = np.random.random(quantity)*(max_c-min_c) + min_c
    A = np.append(c0,A)
    return A 


  def nc_dataframe(): 
    dataframe = pd.DataFrame()

    dataframe['rho'] = Random_n()
    dataframe['VS'] = np.append(Random_c(), np.nan)              #six random values of density of particle (sorted)
    dataframe['dn'] = np.append(0,np.diff(dataframe['rho']))     #six random values of velocity of sound (not sorted)

    return dataframe # Dataframe Columns: {'rho','Velocity of Sound','drho'}

  def NoCrust_dataframe(): #$RandomPart
    dataframe = nc_dataframe()

    n_points = dataframe['rho']
    c_points = dataframe['VS']
    n_diff   = dataframe['dn']


    size = len(n_points) #quantity+2 
    
    E_points = np.empty_like(n_points)
    p_points = np.empty_like(n_points)

    E_points[0] = e0
    p_points[0] = p0

    for i in range (0,size-1):
      ni = n_points[i]
      ei = E_points[i]
      pi = p_points[i]
      ci = c_points[i]

      dn = n_diff[i+1]

      E_points[i+1] = ei + dn*(ei+pi)/ni
      p_points[i+1] = pi + dn*(ei+pi)/ni*ci**2 # %OBS: Eqs (17-18) do artigo 1901.09874

    
    dataframe['e'] = E_points
    dataframe['p'] = p_points
    return dataframe

  dataframe = NoCrust_dataframe()
  
  #$AddingCrust
  del dataframe['dn']  # %OBS: We don't need dn anymore

  dataframe = dataframe[['rho', 'e', 'p', 'VS']]

  crust = crust[:-2]  #Adding the crust (except the last line, which is repeated)
  pd.concat([crust, dataframe], ignore_index=True)

  dataframe = dataframe.drop_duplicates(subset=['rho'])
  dataframe = dataframe.drop_duplicates(subset=['e'])


  dataframe['id'] = model_name



  return dataframe


DATAFRAME = pd.DataFrame()
#############################$Execution#############################
for i in range(N):  
  i_EoS = One_EoS(crust = df, model_name = i, quantity = 6, n_tr  = 0.15 , n_sat =12*0.16, min_c = 0.02, max_c = 0.98)  #%PAR
  DATAFRAME = pd.concat([DATAFRAME, i_EoS], ignore_index=True)
  if i%37==0: 
    print(i)

#############################$Saving#############################
Create_new_file = True
Default_name = str(time.gmtime().tm_year)+ str(time.gmtime().tm_mon) + str(time.gmtime().tm_mday)+ str(time.gmtime().tm_hour) + str(time.gmtime().tm_min) + str(time.gmtime().tm_sec )+ str('.csv')

if Create_new_file == True:
  DATAFRAME.to_csv(Default_name, sep=" ", index= False)

  print("saved to the file: ", Default_name)
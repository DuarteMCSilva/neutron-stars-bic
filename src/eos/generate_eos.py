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

from eos_factory import EquationOfStateFactory

#############################$Input#############################
n = 8  #How many EoS to be generated  (%PAR)
should_create_file = True

data = pd.read_csv('./data/crust.csv')

#############################$Organization#############################
df = pd.DataFrame(data)
df = df.loc[(df['n']<= 0.15)] #%PAR

keys = data.keys()
df = df.rename( {keys[0]: "rho" }, axis = 1) 


#Insert a column for the Velocity of Sound
de = np.append(np.nan,np.diff(df.e))
dp = np.append(np.nan,np.diff(df.p))
cs = np.sqrt(dp/de)
df['VS'] = cs

def get_crust_boundary_conditions(df):
  last_row = df.iloc[-1,]

  print("Last Row Information:")
  print(last_row)

  n0 = last_row['rho']
  e0 = last_row['e']
  p0 = last_row['p']
  c0 = last_row['VS']

  return n0, e0, p0, c0

n0, e0, p0, c0 = get_crust_boundary_conditions(df)
#############################$Execution#############################
dataset = pd.DataFrame()
eos_factory = EquationOfStateFactory(n0, e0, p0, c0, state_transitions_nr = 6)
#############################$Execution#############################
for i in range(n):
  i_EoS = eos_factory.generate(df, i) # $OneEoS
  dataset = pd.concat([dataset, i_EoS], ignore_index=True)
  if i%37==0: 
    print(i)

#############################$Saving#############################
Default_name = str(time.gmtime().tm_year)+ str(time.gmtime().tm_mon) + str(time.gmtime().tm_mday)+ str(time.gmtime().tm_hour) + str(time.gmtime().tm_min) + str(time.gmtime().tm_sec )+ str('.csv')

if should_create_file == True:
  dataset.to_csv(Default_name, sep=" ", index= False)

  print("saved to the file: ", Default_name)
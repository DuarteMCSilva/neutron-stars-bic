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
import numpy as np
import matplotlib.pyplot as plt 
import time 

import os
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
import scipy.integrate as integrate

from multiprocess import Pool # %OBS: There is a similar library called "multiprocessing", 
                                  # might be an option too but exibits some errors which "multiprocess" does not.

#############################$Input#############################
n_cores = 1 #%PAR
data_dir = os.path.join('./data/output_eos/2026-03-21_16-52-25.csv')

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
def TOV_solver_wTD_wroot(EOS_data, N_stars, rho0, rmin, rmax, dr, int_method): #Solves a single EoS
    
    EOS_data.e = EOS_data.e *1.3234e-6 #km-2  #%NOR
    EOS_data.p = EOS_data.p * 1.3234e-6#km-2  #%NOR

    pfloor = min(EOS_data.p) 
    
    rho_max = min(1.5, max(EOS_data.rho)) 
    
    
    def F(r,m,e,p):
        up = r - 4.*np.pi*(r**3)*(e - p)
        down = r - 2.*m
        return up/down

    def B(r,m,e,p,dpde):

        first  = 4.*np.pi*r/(r-2.*m)
        second = - 6./(4.*np.pi*r**2) + 5.*e + 9.*p + (e+p)/dpde
        third  = - 4.*( (m + 4.*np.pi*(r**3)*p)/((r**2)*(1.-2.*m/r)) )**2

        return (first*second + third)

    def k(M,R,yR):
        C = M/R

        first = (8.*C**5)/5. * (1.-2.*C)**2 * (2.+2.*C*(yR - 1.) - yR)
        second = 2.*C*( 6. - 3.*yR + 3.*C*(5.*yR - 8.) )
        third = 4.*C**3*(13. - 11.*yR + C*(3.*yR - 2.) + 2.*C**2*(1. + yR))
        fourth = 3.*(1.- 2.*C)**2 * (2. - yR + 2.*C*(yR - 1.))*(np.log((1.-2*C)))

        return first*((second + third + fourth)**(-1))

    def Lambda(M,R,yR):
        K = k(M,R,yR)
        C = R/M
        return 2./3.*K*C**5


    def found_radius(t,y):
        return (y[1]-pfloor)
    found_radius.terminal = True
    
    rhospan = np.linspace(rho0 , rho_max, N_stars)
    
    rho_ene = IUS(EOS_data.rho, EOS_data.e , k=1)
    rho_pre = IUS(EOS_data.rho, EOS_data.p , k=1)
    pre_ene = IUS(EOS_data.p  , EOS_data.e , k=1)
    ene_pre = IUS(EOS_data.e  , EOS_data.p , k=1)

    dpde_ene = ene_pre.derivative()
    
    integration_interval = (rmin,rmax)

    def TOV(r, y):
        m   = y[0]
        p   = y[1]
        y_i = y[2]
        
        ene  = pre_ene(p)
        dpde = dpde_ene(ene) 
        
        dy    = np.empty_like(y)
        
        #TOV Differential Equations ($TOVeq):
        dy[0] = 4.*np.pi*ene*r**2                     
        dy[1] = -(ene+p)*(m + 4*np.pi*r**3*p)/(r*(r-2*m))  
        dy[2] = - y_i**2/r - y_i*F(r,m,ene,p)/r - r*B(r,m,ene,p,dpde)
        return dy
        
    def solve_ode(y0, dydt_fun, stop_event, verbose=False):

        A = integrate.solve_ivp(dydt_fun , integration_interval , y0
                                ,events = stop_event, first_step = dr
                                ,dense_output=True, method = int_method
                                ,max_step = dr*10, rtol = 1e-9, atol = 1e-12)  #%PAR
        # $OBS: In case " ValueError: f(a) and f(b) must have different signs "
        # changing to a smaller rtol and/or atol might help (might increase computation time)
        return A.t[-1], A.y[:,-1]
    
    def set_initial_conditions(rho, rmin): #($InitialCond)
        p = rho_pre(rho)
        e = rho_ene(rho)
        y = 2.0 #%PAR
        m = 4./3.*np.pi*e*rmin**3
        return m, p, y
    
    R = []
    M = []
    y_R = []

    for rho_i in rhospan:
        
        sol0 = set_initial_conditions(rho_i, rmin)

        t, solu = solve_ode(sol0, TOV, stop_event=found_radius) 

        R.append(t)
        M.append(solu[0])
        y_R.append(solu[2])
    
    R = np.array(R)
    M = np.array(M)
    y_R = np.array(y_R)
    k2 = k(M,R,y_R)
    Lmbda = Lambda(M,R,y_R)
    
    M = M/1.4766 #%NOR
    
    dados = np.array([rhospan, R, M , y_R, k2, Lmbda])
    dados = dados.T
    
    names = ['n','R', 'M', 'y_R', 'k2', 'Lambda']
    
    DataFrame = pd.DataFrame(data = dados,
                             columns = names)
    
    
    ID = EOS_data['id']
    ID = np.array(ID) # %OBS: An arcaic way of picking the id
    DataFrame['id'] = ID[0] 
    
    
    return DataFrame


def Paralelize(DataFrame, ncores):
        
    indices = DataFrame['id'].unique()
    

    def EOS_array(DataFrame): # Creates a list, each element is an EoS
        EOS_list = []

        for ID in indices:
            EOS = DataFrame[DataFrame['id'] == ID]

            EOS_list.append(EOS)
        return EOS_list

    def Solve_for_EOS(EOS):
        return TOV_solver_wTD_wroot(EOS_data = EOS, N_stars = 80, rho0 = 0.05, rmin = 1.e-8 #%PAR
                                    ,rmax=20., dr = 1.e-3,  int_method = 'LSODA')

    
    
    EOS_array = EOS_array(DataFrame)
    
    if(ncores == 1):
        Total_DataFrame = []
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
Final_dataframe = pd.concat(VDF, ignore_index=True)

Create_new_file = True #%PAR
Default_name = str('R_M_L_') + str(time.gmtime().tm_year)+ str(time.gmtime().tm_mon) + str(time.gmtime().tm_mday)+ str(time.gmtime().tm_hour) + str(time.gmtime().tm_min) + str(time.gmtime().tm_sec )+ str('.csv')

if Create_new_file == True:
    Final_dataframe.to_csv(Default_name, index= False) ## sep=" ", 

    print("saved to the file: ", Default_name)





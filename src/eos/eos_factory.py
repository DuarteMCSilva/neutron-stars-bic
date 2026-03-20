import numpy as np
import pandas as pd

class EquationOfStateFactory:
    min_c = 0.02 
    max_c = 0.98
    n_tr  = 0.15
    n_sat = 12*0.16

    def __init__(self, n0, e0, p0, c0, state_transitions_nr = 6):
        self.n0 = n0
        self.e0 = e0
        self.p0 = p0
        self.c0 = c0
        self.quantity = state_transitions_nr

    def generate(self, crust_eos: pd.DataFrame, model_id: int) -> pd.DataFrame:
        core_eos = self.generate_core_eos()
        #$AddingCrust
        crust_eos = crust_eos[:-2]  #Adding the crust (except the last line, which is repeated)
        eos = pd.concat([crust_eos, core_eos], ignore_index=True).drop_duplicates(subset=['rho']).drop_duplicates(subset=['e'])
        eos['id'] = model_id
        return eos

    def generate_core_eos(self):  # $RandomPart: Unknown part - we generate to test against observations
        dataframe = self.generate_basic_properties()

        n_points = dataframe['rho']
        c_points = dataframe['VS']
        n_diff   = dataframe['dn']


        size = len(n_points) #quantity+2 
        
        E_points = np.empty_like(n_points)
        p_points = np.empty_like(n_points)

        E_points[0] = self.e0
        p_points[0] = self.p0

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
        del dataframe['dn']  # %OBS: We don't need dn anymore
        return dataframe[['rho', 'e', 'p', 'VS']]
    
    def generate_basic_properties(self):
        dataframe = pd.DataFrame()

        dataframe['rho'] = self.generate_particle_densities()
        dataframe['VS'] = np.append(self.generate_sound_velocities(), np.nan)              #six random values of density of particle (sorted)
        dataframe['dn'] = np.append(0,np.diff(dataframe['rho']))     #six random values of velocity of sound (not sorted)

        return dataframe # Dataframe Columns: {'rho','Velocity of Sound','drho'}
    
    def generate_sound_velocities(self):
        A = np.random.random(self.quantity)*(self.max_c-self.min_c) + self.min_c
        A = np.append(self.c0,A)
        return A
    
    def generate_particle_densities(self):
        thresholds = np.array([self.n_tr,self.n_sat])
        A = np.random.random(self.quantity)

        n_points = A*(self.n_sat-self.n_tr) + self.n_tr
        n_points = np.append(thresholds,n_points)
        n_points = np.sort(n_points)

        return n_points

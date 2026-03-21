import numpy as np
import pandas as pd

class EquationOfStateFactory:
    '''Factory for generating complete Equations of State (EoS) for neutron stars. 
    This factory takes into account the crust EoS and generates the core EoS based on random properties while ensuring physical consistency.'''

    min_c = 0.02 
    max_c = 0.98
    n_min  = 0.15
    n_sat = 12*0.16

    n_thresholds = np.array([n_min, n_sat])

    def __init__(self, n0, e0, p0, c0, state_transitions_nr = 6):
        self.n0 = n0
        self.e0 = e0
        self.p0 = p0
        self.c0 = c0
        self.quantity = state_transitions_nr

    def generate(self, crust_eos: pd.DataFrame, model_id: int) -> pd.DataFrame:
        '''
            Generates a complete Equation of State by combining the crust EoS with a generated core EoS.

            Parameters:
                - crust_eos:  A DataFrame containing the crust EoS data. It should have columns 'rho', 'e', 'p', and 'VS' representing the density, energy density, pressure, and velocity of sound, respectively.
                
                - model_id: An integer representing the ID of the model.
            Returns:
                - A DataFrame containing the complete EoS data, including both the crust and core. The DataFrame will have columns 'rho', 'e', 'p', 'VS', and 'id', where 'id' corresponds to the model_id parameter.
        '''

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
        '''
            For each state transition, we generate a random sound velocity (in units of c) between physical limits.
            
            **Physical Assumptions**:
                - The sound velocity should be between 0 and 1 (in units of c).
        '''

        A = np.random.random(self.quantity)*(self.max_c-self.min_c) + self.min_c
        A = np.append(self.c0,A)
        return A
    
    def generate_particle_densities(self):
        '''
            For each state transition, we generate a random particle density between physical limits.
            
            **Physical Assumptions**:
                - n is nondecreasing with each transition (increasing depth);
                - n should not exceed n_sat;
                - The first transition occurs at the crust-core boundary (n = n_min).
        '''
        A = np.random.random(self.quantity)

        n_points = A*(self.n_sat-self.n_min) + self.n_min
        n_points = np.append(self.n_thresholds, n_points)
        n_points = np.sort(n_points)

        return n_points

import numpy as np
import pandas as pd
import scipy.integrate as integrate
from scipy.interpolate import InterpolatedUnivariateSpline as IUS

#############################$ToVSolver#############################
class TOV_solver: #Solves a single EoS
    int_method = 'LSODA'
    y = 2.0 # TODO: Find out what is this

    def __init__(self, N_stars = 80, rho0 = 0.05, rmin = 1.e-8, rmax=20., dr = 1.e-3):
        self.N_stars = N_stars
        self.rho0 = rho0
        self.rmin = rmin
        self.rmax = rmax
        self.dr = dr
    
    def solve(self, EOS_data):
        EOS_data.e = EOS_data.e *1.3234e-6 #km-2  #%NOR
        EOS_data.p = EOS_data.p * 1.3234e-6#km-2  #%NOR
        self.pfloor = min(EOS_data.p)
        rho_max = min(1.5, max(EOS_data.rho))
        
        self.rhospan = np.linspace(self.rho0 , rho_max, self.N_stars)
        self.rho_ene = IUS(EOS_data.rho, EOS_data.e , k=1)
        self.rho_pre = IUS(EOS_data.rho, EOS_data.p , k=1)
        self.pre_ene = IUS(EOS_data.p  , EOS_data.e , k=1)
        self.ene_pre = IUS(EOS_data.e  , EOS_data.p , k=1)
        self.dpde_ene = self.ene_pre.derivative()

        self.integration_interval = (self.rmin,self.rmax)
                
        results = self.get_results();
        dados = np.array([self.rhospan] + results)
        dados = dados.T
        
        names = ['n','R', 'M', 'y_R', 'k2', 'Lambda']
        
        DataFrame = pd.DataFrame(data = dados,
                                columns = names)
        
        ID = EOS_data['id']
        ID = np.array(ID) # %OBS: An arcaic way of picking the id
        DataFrame['id'] = ID[0] 
        
        return DataFrame

    def F(self, r,m,e,p):
        up = r - 4.*np.pi*(r**3)*(e - p)
        down = r - 2.*m
        return up/down

    def B(self, r,m,e,p,dpde):

        first  = 4.*np.pi*r/(r-2.*m)
        second = - 6./(4.*np.pi*r**2) + 5.*e + 9.*p + (e+p)/dpde
        third  = - 4.*( (m + 4.*np.pi*(r**3)*p)/((r**2)*(1.-2.*m/r)) )**2

        return (first*second + third)

    def k(self, M, R, yR):
        C = M/R

        first = (8.*C**5)/5. * (1.-2.*C)**2 * (2.+2.*C*(yR - 1.) - yR)
        second = 2.*C*( 6. - 3.*yR + 3.*C*(5.*yR - 8.) )
        third = 4.*C**3*(13. - 11.*yR + C*(3.*yR - 2.) + 2.*C**2*(1. + yR))
        fourth = 3.*(1.- 2.*C)**2 * (2. - yR + 2.*C*(yR - 1.))*(np.log((1.-2*C)))

        return first*((second + third + fourth)**(-1))

    def Lambda(self, M, R, yR):
        K = self.k(M,R,yR)
        C = R/M
        return 2./3.*K*C**5

    def TOV(self,r, y):
        m   = y[0]
        p   = y[1]
        y_i = y[2]
        
        ene  = self.pre_ene(p)
        dpde = self.dpde_ene(ene) 
        
        dy    = np.empty_like(y)
        
        #TOV Differential Equations ($TOVeq):
        dy[0] = 4.*np.pi*ene*r**2                     
        dy[1] = -(ene+p)*(m + 4*np.pi*r**3*p)/(r*(r-2*m))  
        dy[2] = - y_i**2/r - y_i*self.F(r,m,ene,p)/r - r*self.B(r,m,ene,p,dpde)
        return dy
        
    def solve_ode(self, y0, dydt_fun, verbose=False):
        floor = self.pfloor
        def _found_radius(t,y):
            return (y[1]-floor)
        _found_radius.terminal = True


        A = integrate.solve_ivp(dydt_fun , self.integration_interval , y0
                                ,events = _found_radius, first_step = self.dr
                                ,dense_output=True, method = self.int_method
                                ,max_step = self.dr*10, rtol = 1e-9, atol = 1e-12)  #%PAR
        # $OBS: In case " ValueError: f(a) and f(b) must have different signs "
        # changing to a smaller rtol and/or atol might help (might increase computation time)
        return A.t[-1], A.y[:,-1]
    
    def set_initial_conditions(self, rho, rmin): #($InitialCond)
        p = self.rho_pre(rho)
        e = self.rho_ene(rho)
        y = self.y
        m = 4./3.*np.pi*e*rmin**3
        return m, p, y
    
    def get_results(self):
        R = []
        M = []
        y_R = []

        for rho_i in self.rhospan:
            
            sol0 = self.set_initial_conditions(rho_i, self.rmin)

            t, solu = self.solve_ode(sol0, self.TOV)
            R.append(t)
            M.append(solu[0])
            y_R.append(solu[2])
        
        R = np.array(R)
        M = np.array(M)
        y_R = np.array(y_R)
        k2 = self.k(M,R,y_R)
        Lmbda = self.Lambda(M,R,y_R)
        M = M/1.4766 #%NOR
        return [R, M, y_R, k2, Lmbda]

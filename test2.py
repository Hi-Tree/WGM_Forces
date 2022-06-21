import numpy as np
from numpy.random.mtrand import rand
import forces
from scipy.integrate import odeint
from scipy.stats import maxwell 
import matplotlib.pyplot as plt
import pandas as pd

lambd = 2*np.pi*39.107/(5.6e-5)
p = 1040 
M_avg = 4/3 * np.pi * p* (7.5e-7)**3 
tao = 0.00048528
kbt = 1.38064852e-23 * 300 
vavg = (4*kbt) / (3*M_avg)
vavgTilda = tao*vavg/lambd #1040kg/m^3
j = 0
TC = []
eta = (0.00700175)/(vavgTilda*6*np.pi*(7.5*10**(-7)))
for i in range(1):
    N = 10
    Results = pd.DataFrame(columns=['rand_x','rand_y','rand_z','rho_','theta_','phi_','dRho_','dTheta_','dPhi_','Rp_','gamma_','time_'])
    while Results.count()[0] != N:
        Rp = np.random.uniform(7.5e-7 - 3.75e-8, 7.5e-7 + 3.75e-8)
        M = 4/3 * np.pi * p* Rp**3 #specific mass
        rho = 41 #np.random.uniform(39.107,39.107+(2*np.pi))
        theta = np.pi/2 #np.random.uniform((np.pi-np.sqrt(1/40)),(np.pi+np.sqrt(1/40)))
        phi = 0 #np.random.uniform(0,np.pi*2)
        vx = np.random.normal(0,vavgTilda)
        vy = np.random.normal(0,vavgTilda)
        vz = np.random.normal(0,vavgTilda)
        gamma = (9*tao*eta)/(2*Rp*rho)
        dRhodT = vx*np.sin(theta)*np.cos(phi)+vy*np.sin(theta)*np.sin(phi)+vz*np.cos(theta)
        if dRhodT <= 0:
            Results = pd.concat([pd.DataFrame([[vx,vy,vz,rho,theta,phi,dRhodT,0,0,Rp,gamma,None]],columns = Results.columns),Results],ignore_index = True)


print(Results)


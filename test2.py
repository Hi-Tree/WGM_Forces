import numpy as np
import forces
from scipy.integrate import odeint
from scipy.stats import maxwell 
import matplotlib.pyplot as plt
import pandas as pd


def randomized(N):
    Results = pd.DataFrame(columns=['rand_x','rand_y','rand_z','rho_','theta_','phi_','gamma_','time_'])
    while Results.count()[0] != N:
        Rp = np.random.normal(loc = 7.5*10**(-7), scale = 3.75*10**(-8))
        rho = np.random.uniform(39.107,39.107+(2*np.pi))
        theta = np.random.uniform((np.pi-np.sqrt(1/40)),(np.pi+np.sqrt(1/40)))
        phi = np.random.uniform(0,np.pi*2)
        vx = maxwell.rvs()
        vy = maxwell.rvs()
        vz = maxwell.rvs()
        eta = (0.00700175)/(vx*6*np.pi*Rp)
        gamma = 6*np.pi*eta*Rp 
        dRhodT = vx*np.sin(theta)*np.cos(phi)+vy*np.sin(theta)*np.sin(phi)+vz*np.cos(theta)
        Results = Results.append([pd.DataFrame([[vx,vy,vz,rho,theta,phi,gamma,None]],columns = Results.columns),Results])
    return Results

m = randomized(5)

print(m)
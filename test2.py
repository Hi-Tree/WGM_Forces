import numpy as np
from numpy.random.mtrand import rand
import forces
from scipy.integrate import odeint
from scipy.stats import maxwell 
import matplotlib.pyplot as plt
import pandas as pd

#keep track of brownian motion 
TC = []
RhoP =[]
ThetaP = []
PhiP = []

def randomized(N):
    Results = pd.DataFrame(columns=['rand_x','rand_y','rand_z','rho_','theta_','phi_','dRho_','dTheta_','dPhi_','Rp_','gamma_','time_'])
    while Results.count()[0] != N:
        Rp = np.random.uniform(7.5*10**(-7)-3.75*10**(-8), 7.5*10**(-7)+3.75*10**(-8))
        rho = np.random.uniform(39.107,39.107+(2*np.pi))
        theta = np.random.uniform((np.pi-np.sqrt(1/40)),(np.pi+np.sqrt(1/40)))
        phi = np.random.uniform(0,np.pi*2)
        vx = maxwell.rvs() #change later
        vy = maxwell.rvs()
        vz = maxwell.rvs()
        eta = (0.00700175)/(vx*6*np.pi*Rp)
        gamma = 6*np.pi*eta*Rp 
        dRhodT = vx*np.sin(theta)*np.cos(phi)+vy*np.sin(theta)*np.sin(phi)+vz*np.cos(theta)
        if dRhodT <= 0.0075:
            Results = pd.concat([pd.DataFrame([[vx,vy,vz,rho,theta,phi,dRhodT,np.pi/2,1e-10,Rp,gamma,None]],columns = Results.columns),Results],ignore_index = True)
    return Results

def f(u, t, par, gamma):
    rho, drho, theta, dtheta, phi, dphi = u
    dudt = [
            drho, forces.rho(rho, theta, par) - gamma * drho + rho * (dtheta * np.cos(phi)) ** 2 + rho * dphi ** 2,
            dtheta, (forces.theta(rho, theta, par) - gamma * rho * dtheta * np.cos(phi) - 2 * drho * dtheta * np.cos(phi) + 2 * rho * dtheta * dphi * np.sin(phi)) / (rho * np.cos(phi)),
            dphi, (forces.phi(rho, theta, par) - gamma * rho * dphi - 2 * drho * dphi - rho * dphi ** 2 * np.sin(phi) * np.cos(phi)) / rho
        ]
    if rho >= 39.107 and t >= 12: #if time is greater than 12 and we have not yet reached resonator change V
        TC.append(t)
        RhoP.append(rho)
        ThetaP.append(theta)
        PhiP.append(phi)
        return [None,None,None,None,None,None]
    if rho <= 39.107: #if we reached the resonator just save time
        TC.append(t)
        return [None,None,None,None,None,None]
    return dudt

def brownianMotion(df):
    n = randomized(1)
    n.loc[0,'rho_'] = df.loc[0,'rho_']
    n.loc[0,'theta_'] = df.loc[0,'rho_']
    n.loc[0,'phi_'] = df.loc[0,'rho_']
    n.loc[0,'Rp_'] = df.loc[0,'rho_']
    n.loc[0,'gamma'] = df.loc[0,'rho_']

M = randomized(1) # just returns what to potentially try
j = 0
time = np.linspace(0, 100, 1000) #original time, time resets when calling equation continuity done manually 
while j != M.shape[0]:
    gamma_ = M['gamma_'][j]
    par = {
        'l': 40,
        'x_r': 39.107,            
        'R_ratio': (M['Rp_'][j])/(5.6*10**(-5)), 
        'n_w': 1.326,
        'n_p': 1.572
        }
    u0 = [M['rho_'][j],M['dRho_'][j], M['theta_'][j], M['dTheta_'][j], M['phi_'][j], M['dPhi_'][j]] 
    sol = odeint(f, u0, time, args = (par, gamma_*0))
    if len(TC) != 0:
        M.at[j,'time_'] = TC[0]
    TC.clear()
    j+=1


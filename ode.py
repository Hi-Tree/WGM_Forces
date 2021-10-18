import numpy as np
import forces
from scipy.integrate import odeint
from scipy.stats import maxwell 
import matplotlib.pyplot as plt
import pandas as pd

################  Changes Made ##############
#dtheta = v_x*np.cos(theta)*np.cos(phi)+v_y*np.cos(theta)*np.sin(phi)-v_z*np.sin(theta) 
# #dphiv_y*np.cos(phi)-v_x*np.sin(phi)
# randomize Rp mean = 750nm with 5% deviation 
# adjust eta such that v * gamma = || 0.00700175 = v * 6*np.pi*eta*Rp
# uniform random positions
# purely data collective no graphs display, quits uppon reaching resonator 
# ****** IMP if pos1 > pos2 and not reached resonator quit
# Manually Found that if dRho/dt > 0.0075 particle will not attach 
j = 0
TC = []
###################################################    RANDOMIZE ALL VARIABLES #####################################################
N = 5
Results = pd.DataFrame(columns=['rand_x','rand_y','rand_z','rho_','theta_','phi_','dRho_','dTheta_','dPhi_','Rp_','gamma_','time_'])
while Results.count()[0] != N:
    Rp = np.random.uniform(7.5*10**(-7)-3.75*10**(-8), 7.5*10**(-7)+3.75*10**(-8))
    rho = np.random.uniform(39.107,39.107+(2*np.pi))
    theta = np.random.uniform((np.pi-np.sqrt(1/40)),(np.pi+np.sqrt(1/40)))
    phi = np.random.uniform(0,np.pi*2)
    vx = maxwell.rvs()
    vy = maxwell.rvs()
    vz = maxwell.rvs()
    eta = (0.00700175)/(vx*6*np.pi*Rp)
    gamma = 6*np.pi*eta*Rp 
    dRhodT = vx*np.sin(theta)*np.cos(phi)+vy*np.sin(theta)*np.sin(phi)+vz*np.cos(theta)
    if dRhodT <= 0.0075:
        Results = pd.concat([pd.DataFrame([[vx,vy,vz,rho,theta,phi,dRhodT,0,0,Rp,gamma,None]],columns = Results.columns),Results],ignore_index = True)

############################################################  Function #######################################################################################
def f(u, t, par, gamma):
    rho, drho, theta, dtheta, phi, dphi = u
    dudt = [
            drho, forces.rho(rho, theta, par) - gamma * drho + rho * (dtheta * np.cos(phi)) ** 2 + rho * dphi ** 2,
            dtheta, (forces.theta(rho, theta, par) - gamma * rho * dtheta * np.cos(phi) - 2 * drho * dtheta * np.cos(phi) + 2 * rho * dtheta * dphi * np.sin(phi)) / (rho * np.cos(phi)),
            dphi, (forces.phi(rho, theta, par) - gamma * rho * dphi - 2 * drho * dphi - rho * dphi ** 2 * np.sin(phi) * np.cos(phi)) / rho
        ]
    if rho <= 39.107:
        TC.append(t)
    return dudt
###################################################### Parameters and Solutions #######################################################################

time = np.linspace(0, 80, 100)
while j != N:
    gamma_ = Results['gamma_'][j]
    par = {
        'l': 40,
        'x_r': 39.107,            
        'R_ratio': (Results['Rp_'][j])/(5.6*10**(-5)), 
        'n_w': 1.326,
        'n_p': 1.572
        }
    u0 = [Results['rho_'][j],Results['dRho_'][j], Results['theta_'][j], Results['dTheta_'][j], Results['phi_'][j], Results['dPhi_'][j]] 
    sol = odeint(f, u0, time, args = (par, gamma*0))
    Results.at[j,'time_'] = TC[0]
    TC.clear()
    j+=1

####################################################### Viewing Options ########################################################
pd.set_option('display.max_colwidth', None)
print(Results)
#### End of Data Collection

#v = sphere 750nm rad 
#Graph
'''
fig = plt.figure()
gs = fig.add_gridspec(3, hspace = 0)
axs = gs.subplots(sharex = True)
axs[0].plot(time, sol[:, 0], markersize = 2)
#show resonator's surface
axs[0].axhline(par['x_r'], ls = 'dashed', alpha = 0.5, color = 'red')
axs[0].set_ylabel(r'$\rho$')
axs[1].plot(time, sol[:, 2], markersize = 2)
axs[1].set_ylabel(r'$\theta$')
axs[2].plot(time, sol[:, 4], markersize = 2)
axs[2].set_ylabel(r'$\phi$')

for ax in axs:
    ax.label_outer()
    
plt.xlabel('Time')
plt.show()
'''
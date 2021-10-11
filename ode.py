import numpy as np
import forces
from scipy.integrate import odeint
from scipy.stats import maxwell 
import matplotlib.pyplot as plt
import pandas as pd

################  Changes Made ##############
# randomize Rp mean = 750nm with 5% deviation 
# adjust eta such that v * gamma = || 0.00700175 = v * 6*np.pi*eta*Rp
# uniform random positions
# purely data collective no graphs display, quits uppon reaching resonator 
# ****** IMP if pos1 > pos2 and not reached resonator quit
N = 1
Results = pd.DataFrame()
def randomized(vx_max):
    rand_x = []
    rand_y = []
    rand_z = []
    while len(rand_x) != N:
        vx = maxwell.rvs()
        vy = maxwell.rvs()
        vz = maxwell.rvs()
        if vx < vx_max: #less than max and positive
            rand_x.append(0.01)
            rand_y.append(0.01)
            rand_z.append(0.01)
    rand_v = np.zeros((N,3),dtype = float)
    rand_v[:,0], rand_v[:,1], rand_v[:,2] = [rand_x,rand_y,rand_z]
    return rand_v 

#Initial Positions
rho = 39.107*(2 * np.pi) * 1.1 #because of deffinitions x_r is manually entered 
theta = np.pi / 2
phi = 0

#Initial Velocities 
values = randomized(.5)        
v_x = values[:,0]
v_y = values[:,1]
v_z = values[:,2] 
drho = v_x*np.sin(theta)*np.cos(phi)+v_y*np.sin(theta)*np.sin(phi)+v_z*np.cos(theta)
dtheta = v_x*np.cos(theta)*np.cos(phi)+v_y*np.cos(theta)*np.sin(phi)-v_z*np.sin(theta)
dphi = v_y*np.cos(phi)-v_x*np.sin(phi)

Rp = np.random.normal(loc = 7.5*10**(-7), scale = 3.75*10**(-8), size = N)
eta = [(0.00700175)/(v_x[i]*6*np.pi*Rp[i]) for i in range(N)]
gamma_ = [6*np.pi*eta[i]*Rp[i] for i in range(N)]

#v = sphere 750nm rad
allData = []
time = np.linspace(0, 80, 100)

def f(u, t, par, gamma):
    rho, drho, theta, dtheta, phi, dphi = u
    dudt = [
            drho, forces.rho(rho, theta, par) - gamma * drho + rho * (dtheta * np.cos(phi)) ** 2 + rho * dphi ** 2,
            dtheta, (forces.theta(rho, theta, par) - gamma * rho * dtheta * np.cos(phi) - 2 * drho * dtheta * np.cos(phi) + 2 * rho * dtheta * dphi * np.sin(phi)) / (rho * np.cos(phi)),
            dphi, (forces.phi(rho, theta, par) - gamma * rho * dphi - 2 * drho * dphi - rho * dphi ** 2 * np.sin(phi) * np.cos(phi)) / rho
        ]
    return dudt

#Solve
for j in range(N):
    gamma = gamma_[j]
    par = {
        'l': 40,
        'x_r': 39.107,            
        'R_ratio': (Rp[j])/(5.6*10**(-5)), 
        'n_w': 1.326,
        'n_p': 1.572
        }
    u0 = [rho, drho[j], theta, dtheta[j], phi, dphi[j]] 
    sol = odeint(f, u0, time, args = (par, gamma*0))
    
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
import numpy as np
import forces
from scipy.integrate import odeint
from scipy.stats import maxwell 
import matplotlib.pyplot as plt

################  Changes Made ##############
# randomize Rp mean = 750nm with 5% deviation 
# adjust eta such that v * gamma = || 0.00700175 = v * 6*np.pi*eta*Rp
N = 1

def randomized(vx_max):
    rand_x = []
    rand_y = []
    rand_z = []
    while len(rand_x) != N:
        vx = maxwell.rvs()
        vy = maxwell.rvs()
        vz = maxwell.rvs()
        if vx < vx_max and vx >= 0 : #less than max and positive
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
time = np.linspace(0, 50, 100)

def f(u, t, par, gamma):
    rho, drho, theta, dtheta, phi, dphi = u
    if rho >= par['x_r']: #Stop when reach resonator
        dudt = [
            drho, forces.rho(rho, theta, par) - gamma * drho + rho * (dtheta * np.cos(phi)) ** 2 + rho * dphi ** 2,
            dtheta, (forces.theta(rho, theta, par) - gamma * rho * dtheta * np.cos(phi) - 2 * drho * dtheta * np.cos(phi) + 2 * rho * dtheta * dphi * np.sin(phi)) / (rho * np.cos(phi)),
            dphi, (forces.phi(rho, theta, par) - gamma * rho * dphi - 2 * drho * dphi - rho * dphi ** 2 * np.sin(phi) * np.cos(phi)) / rho
        ]
        #print(rho)
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
    sol = odeint(f, u0, time, args = (par, gamma))
    m,b = np.polyfit(time,sol[:,0],1) #linear fit we just want to see if it is dicreasing
    print(m)
    if m < 0: #if the line of best fit for rho is negative 
        allData.append(u0) # save the initial conditions
        print(Rp[j])
print(allData)
#Graph

fig, (ax1,ax2) = plt.subplots(2)
ax1.plot(time,(m*time)+b)
ax1.plot(time,sol[:,0])
ax2.plot(time,sol[:,1])
plt.show()
'''
fig, (ax1,ax2, ax3,ax4,ax5,ax6) = plt.subplots(6)
ax1.plot(time, sol[:, 0], "-", markersize = 2)
ax1.set_ylabel("rho")
ax2.plot(time, sol[:, 1], "-", markersize = 2)
ax2.set_ylabel("drho")
ax3.plot(time, sol[:, 2], "-", markersize = 2)
ax3.set_ylabel("theta")
ax4.plot(time, sol[:, 3], "-", markersize = 2)
ax4.set_ylabel("dtheta")
ax5.plot(time, sol[:, 4], "-", markersize = 2) 
ax5.set_ylabel("phi")
ax6.plot(time, sol[:, 5], "-", markersize = 2)
ax6.set_ylabel("dphi")
plt.xlabel("time")
plt.show()
'''
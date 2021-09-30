import forces
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


time = np.linspace(0, 80000, 10000)

def f(u, t, par, gamma):
    rho, drho, theta, dtheta, phi, dphi = u
    dudt = [
            drho, forces.rho(rho, theta, par) - gamma * drho + rho * (dtheta * np.cos(phi)) ** 2 + rho * dphi ** 2,
            dtheta, (forces.theta(rho, theta, par) - gamma * rho * dtheta * np.cos(phi) - 2 * drho * dtheta * np.cos(phi) + 2 * rho * dtheta * dphi * np.sin(phi)) / (rho * np.cos(phi)),
            dphi, (forces.phi(rho, theta, par) - gamma * rho * dphi - 2 * drho * dphi - rho * dphi ** 2 * np.sin(phi) * np.cos(phi)) / rho
        ]
    return dudt

v_x = 0.00000000000001
v_y = 0.01
v_z = 0.001

rho = 40
theta = np.pi / 2
phi = 0

drho = v_x*np.sin(theta)*np.cos(phi)+v_y*np.sin(theta)*np.sin(phi)+v_z*np.cos(theta)
dtheta = v_x*np.cos(theta)*np.cos(phi)+v_y*np.cos(theta)*np.sin(phi)-v_z*np.sin(theta)
dphi = v_y*np.cos(phi)-v_x*np.sin(phi)

par = {
        'l': 40,
        'x_r': 39.107,            
        'R_ratio': 7.5e-2 / 5.6, 
        'n_w': 1.326,
        'n_p': 1.572
        }

eta = (0.00700175)/(v_x*6*np.pi*7.5e-2)
gamma =  6*np.pi*eta*7.5e-2

u0 = [rho, drho, theta, dtheta, phi, dphi]
sol = odeint(f, u0, time, args = (par, gamma))





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

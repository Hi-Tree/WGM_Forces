import forces
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


time = np.linspace(0, 100, 100)

def f(u, t, par, gamma): #correct gamma tilde subs ****
    rho, drho, theta, dtheta, phi, dphi = u
    dudt = [
            drho, forces.rho(rho, theta, par) - gamma * drho + rho * (dtheta * np.cos(phi)) ** 2 + rho * dphi ** 2,
            dtheta, (forces.theta(rho, theta, par) - gamma * rho * dtheta * np.cos(phi) - 2 * drho * dtheta * np.cos(phi) + 2 * rho * dtheta * dphi * np.sin(phi)) / (rho * np.cos(phi)),
            dphi, (forces.phi(rho, theta, par) - gamma * rho * dphi - 2 * drho * dphi - rho * dphi ** 2 * np.sin(phi) * np.cos(phi)) / rho
        ]
    if rho <= 39.107:
        print("we did it")
    return dudt

rho = 41
theta = np.pi/2
phi = 0

drho = 0
dtheta = 0
dphi = 0
# find how they are related 
par = {
        'l': 40,
        'x_r': 39.107,            
        'R_ratio': (7.125e-2 + 3.75e-3)/ 5.6, 
        'n_w': 1.326,
        'n_p': 1.572
        }


gamma =  0

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

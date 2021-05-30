import numpy as np
import forces
from scipy.integrate import odeint
import matplotlib.pyplot as plt

v_x = 0.001
v_y = 0.001 
v_z = 0

gamma = 2
params = {
  'l': 720,
  'x_r': 524.5,            # resonance k0 * R
  'R_ratio': 7.5e-2 / 5.6, # R_p / R_r
  'n_w': 1.326,
  'n_p': 1.572
}

def f(u, t, params, gamma):
    rho, drho, theta, dtheta, phi, dphi = u
    if rho >= 1.5:
        dudt = [
            drho, forces.rho(rho, theta, params) - gamma * drho + rho * (dtheta * np.cos(phi)) ** 2 + rho * dphi ** 2,
            dtheta, (forces.theta(rho, theta, params) - gamma * rho * dtheta * np.cos(phi) - 2 * drho * dtheta * np.cos(phi) + 2 * rho * dtheta * dphi * np.sin(phi)) / (rho * np.cos(phi)),
            dphi, (forces.phi(rho, theta, params) - gamma * rho * dphi - 2 * drho * dphi - rho * dphi ** 2 * np.sin(phi) * np.cos(phi)) / rho
        ]
    return dudt

time = np.linspace(0, 1.25, 100)

# initial conditions
rho = params['x_r'] / (2 * np.pi) * 1.5 # 1.5 of resonator's radius

theta = np.pi / 2

phi = 0.001 #np.pi / 2 * 0 #just the initial position is zero but it changes with time

drho = v_x*np.sin(theta)*np.cos(phi)+v_y*np.sin(theta)*np.sin(phi)+v_z*np.cos(theta)
dtheta = v_x*np.cos(theta)*np.cos(phi)+v_y*np.cos(theta)*np.sin(phi)-v_z*np.sin(theta)
dphi = v_y*np.cos(phi)-v_x*np.sin(phi)

u0 = [rho, drho, theta, dtheta, phi, dphi]
sol = odeint(f, u0, time, args = (params, gamma))


fig = plt.figure()
gs = fig.add_gridspec(3, hspace = 0)
axs = gs.subplots(sharex = True)
axs[0].plot(time, sol[:, 0], "o", markersize = 2)
# show resonator's surface
#axs[0].axhline(params['x_r'] / (2 * np.pi), ls = 'dashed', alpha = 0.5, color = 'red')
axs[0].set_ylabel(r'$\rho$')
axs[1].plot(time, sol[:, 2] % np.pi, "o", markersize = 2)
axs[1].set_ylabel(r'$\theta$')
axs[2].plot(time, sol[:, 4] % (2 * np.pi), "o", markersize = 2)
axs[2].set_ylabel(r'$\phi$')

for ax in axs:
    ax.label_outer()
    
plt.xlabel('Time')
plt.show()

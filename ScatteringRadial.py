from scipy.special import spherical_jn, spherical_yn
import numpy as np
import matplotlib.pyplot as plt

l = 720
n_w = 1.326
n_p = 1.572
X_r = 524.5
R_r = 5.6*10**(-5)
R_p = 7.5*10**(-7)


def f_sc_r(x,theta):
    f = (2/3)*n_w*((X_r**3)/(x**2))*((R_p**3)/(R_r**3))*\
        (((n_p**2)-(n_w**2))/((n_p**2)+2*(n_w**2)))
    sinf = np.sin(theta)**(2*l-2)
    cosf = 1+np.cos(theta)**2
    bessels = 1/(((spherical_jn(l,n_w*X_r)*spherical_jn(l,n_w*X_r,True))\
                   +(spherical_yn(l,n_w*X_r)*spherical_yn(l,n_w*X_r,True))))
    final = f*sinf*cosf*bessels
    return final

g = np.linspace(X_r,X_r*2,500)
y = f_sc_r(g,np.pi/2)
plt.plot(g,y)
plt.show()

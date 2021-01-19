from scipy.special import spherical_jn, spherical_yn
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

l = 720
n_w = 1.326
n_p = 1.572
X_r = 524.5
R_r = 5.6*10**(-5)
R_p = 7.5*10**(-7)
k = X_r/R_r


def f_sc_phi(theta,x):
    f = (2/3)*(n_w**2)*((X_r**3)/x)*((R_p**3)/(R_r**3))*(((n_p**2)-(n_w**2))/((n_p**2)+(2*(n_w**2))))
    g = np.sin(theta)**(2*l-3)
    numerator = (l*(1+np.sin(theta))- np.cos(2*theta))\
                *((spherical_jn(l,n_w*x)*spherical_jn(l,n_w*x))+(spherical_yn(l,n_w*x)*spherical_yn(l,n_w*x)))
    denominator = ((spherical_jn(l,n_w*X_r)*spherical_jn(l,n_w*X_r,True))\
                   +(spherical_yn(l,n_w*X_r)*spherical_yn(l,n_w*X_r,True)))
    m = numerator/denominator
    final = f*g*m
    return final

def f_sc_rho(theta,x):
    f = (2/3)*n_w*((X_r**3)/x**2)*((R_p**3)/(R_r**3))*(((n_p**2)-(n_w**2))/((n_p**2)+(2*(n_w**2))))
    g = sp.sin(theta)**(2*l-3)
    numerator = (1+(sp.cos(theta)**2))
    denominator = ((spherical_jn(l,n_w*X_r)*spherical_jn(l,n_w*X_r,True))\
                   +(spherical_yn(l,n_w*X_r)*spherical_yn(l,n_w*X_r,True)))
    final = f*g*numerator/denominator
    return final



'''  
g = np.linspace(X_r,X_r*1.01,500)
y = f_sc_sum(np.pi/2,g)
plt.plot(g,y)
plt.show()
'''


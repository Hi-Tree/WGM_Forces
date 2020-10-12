from scipy.special import spherical_jn, spherical_yn
import numpy as np
import matplotlib.pyplot as plt

l = 720
n_w = 1.326
n_p = 1.572
X_r = 524.5
R_r = 5.6*10**(-5)
R_p = 7.5*10**(-7)
k = X_r/R_r


def f_sc_phi(x,theta):
    f = (2/3)*(n_w**2)*((X_r**3)/x)*((R_p**3)/(R_r**3))*(((n_p**2)-(n_w**2))/((n_p**2)+(2*(n_w**2))))
    g = np.sin(theta)**(2*l-3)
    numerator = (l*(1+np.sin(theta))- np.cos(2*theta))\
                *((spherical_jn(l,n_w*x)*spherical_jn(l,n_w*x))+(spherical_yn(l,n_w*x)*spherical_yn(l,n_w*x)))
    denominator = ((spherical_jn(l,n_w*X_r)*spherical_jn(l,n_w*X_r,True))\
                   +(spherical_yn(l,n_w*X_r)*spherical_yn(l,n_w*X_r,True)))
    m = numerator/denominator
    final = f*g*m
    return final

g = np.linspace(X_r,X_r*1.01,500)
y = f_sc_phi(g,np.pi/2)
plt.plot(g,y)
plt.show()
    
    

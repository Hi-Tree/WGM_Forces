import numpy as np
from scipy.special import spherical_yn, spherical_jn
import matplotlib.pyplot as plt

l = 720
n_w = 1.326
X_r = 524.5

#define dimensionless polar component
#X_r is radius, x is variable 
def f_gTheta(theta,x):
    bessel1 = (spherical_jn(l,n_w*x)*spherical_jn(l,n_w*x)) + \
              (spherical_yn(l,n_w*x)*spherical_yn(l,n_w*x))
    bessel2 = ((spherical_yn(l,n_w*X_r)*spherical_yn(l,n_w*X_r,True)) + \
              (spherical_yn(l,n_w*X_r)*spherical_yn(l,n_w*X_r,True)))*n_w*x
    bessels = bessel1/bessel2
    rest = (np.sin(theta)**(2*l-3))*((l-1)*(1+(np.cos(theta)**2)) \
                                     -((np.sin(theta)**2)*np.cos(theta)))
    final = rest*bessels
    return final

#Plot values for angle constant

n = np.linspace(X_r,X_r*1.01,500)
xx = n.tolist()

ys = []
for i in range(0,500,1):
    y = f_gTheta(np.pi/2,xx[i])
    ys.append(y)

yy = []
for h in ys:
    yy.append(float(h))

plt.plot(xx,yy)
plt.show()

'''

#Plot values for distance constant
g = np.linspace(np.pi/4,3*np.pi/2,500)
y = f_gTheta(g,X_r)

plt.plot(g,y)
plt.show()
'''


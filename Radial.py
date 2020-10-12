import numpy as np
from scipy.special import spherical_jn
from scipy.special import spherical_yn
import sympy as sp
import matplotlib.pyplot as plt
R_r = 5.6*10**(-5) 
l = 720
n_w = 1.326
#k = 524.5/R_r
X_r = 524.5



# R is constant r is changing  
def f_gr(theta,x):
    f = ((sp.sin(theta))**(2*l-2))*(1+(sp.cos(theta))**2)
    b = (spherical_jn(l,n_w*x)*spherical_jn(l,n_w*x,True))+(spherical_yn(l,n_w*x)*spherical_yn(l,n_w*x,True))
    c = (spherical_jn(l,n_w*X_r)*spherical_jn(l,n_w*X_r,True))+(spherical_yn(l,n_w*X_r)*spherical_yn(l,n_w*X_r,True))
    n = b/c
    f = f*n
    return f

print(f_gr(np.pi/2,X_r))

n = np.linspace(X_r,1.01*X_r,500)
zz = n.tolist()
#[Decimal(i) for i in z]

#print(z)
ys = []
for i in range(0,500,1):
    y = f_gr(np.pi/2,zz[i])
    ys.append(y)

yy = []
for h in ys:
    yy.append(float(h))
#print(yy,"cat")

plt.plot(zz,yy)
plt.show()
print(f_gr(sp.pi/2, 1.0001))



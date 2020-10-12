import numpy as np
from scipy.special import spherical_yn, spherical_jn
import decimal 
import factorials #made by me

#define real part of alpha
def Real_alpha(R_p, n_p, n_w):
    R = 4*(np.pi)*(R_p**3)*(((n_p**2)-(n_w**2))/((n_p**2)+(n_w**2)))
    return R

#define gamma
def gamma(l):
    m = (2*l)+1
    n = 4*(np.pi)*l*(l+1)*factorials.factorial2(2*l-1)*(2**l)*factorials.factorial(l)
    g = np.sqrt((m/n))
    return g

print(gamma(5))

 

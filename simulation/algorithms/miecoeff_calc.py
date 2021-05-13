"""
--> The following code for calculating and evaluating holographic images
    in OpenCL is based on the ideas and developement of GRIER D.

--> mie_f function, received from THALHAMMER G.
"""
# import of necessary modules and functions
import numpy as np

from scipy.special import riccati_jn, riccati_yn

# theory function that calculates scattering arrays
def mie_f(r, n2, lamda, n_medium, Nmax):
    k = 2*np.pi/lamda
    m = n2/n_medium
    x = k*r*n_medium

    psi_x, psi_x_d = riccati_jn(Nmax, x)
    psi_mx, psi_mx_d = riccati_jn(Nmax, m*x)

    chi_x, chi_x_d = riccati_yn(Nmax, x)
    chi_mx, chi_mx_d = riccati_yn(Nmax, m*x)

    xi_x, xi_x_d = (psi_x + 1j*chi_x, psi_x_d + 1j*chi_x_d)

    Ta = (m*psi_mx*psi_x_d - psi_x*psi_mx_d)/(m*psi_mx*xi_x_d - xi_x*psi_mx_d)
    Tb = (psi_mx*psi_x_d - m*psi_x*psi_mx_d)/(psi_mx*xi_x_d - m*xi_x*psi_mx_d)

    Ta[np.isnan(Ta)]=0.
    Tb[np.isnan(Tb)]=0.
    return Ta.astype(np.complex64), Tb.astype(np.complex64)
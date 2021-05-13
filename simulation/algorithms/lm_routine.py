"""
--> LM function, received from THALHAMMER G.
"""

# import necessary modules and functions
import numpy as np

# Implementation of the Levenberg-Marquardt algorithm in pure
# Python. Solves the normal equations.
def LM(fun, pars, args,
   tau = 1e-2, eps1 = 1e-6, eps2 = 1e-6, kmax = 20,
   verbose = False,
   full_output = False):

    p = pars
    f, J = fun(p, *args)

    A = np.inner(J,J)
    g = np.inner(J,f)
    I = np.eye(len(p))

    k = 0; nu = 2
    mu = tau * max(np.diag(A))
    stop = np.linalg.norm(g, np.Inf) < eps1
    while not stop and k < kmax:
        k += 1
        try:
            d = np.linalg.solve( A + mu*I, -g)
        except np.linalg.LinAlgError:
            print("Singular matrix encountered in LM")
            stop = True
            reason = 'singular matrix'
            break

        if np.linalg.norm(d) < eps2*(np.linalg.norm(p) + eps2):
            stop = True
            reason = 'small step'
            break
        pnew = p + d

        fnew, Jnew = fun(pnew, *args)
        rho = (np.linalg.norm(f)**2 - np.linalg.norm(fnew)**2)/np.inner(d, mu*d - g)

        if rho > 0:
            p = pnew
            A = np.inner(Jnew, Jnew)
            g = np.inner(Jnew, fnew)
            f = fnew
            J = Jnew
            if (np.linalg.norm(g, np.Inf) < eps1):
                stop = True
                reason = "small gradient"
                break
            mu = mu * max([1.0/3, 1.0 - (2*rho - 1)**3])
            nu = 2.0
        else:
            mu = mu * nu
            nu = 2*nu

        if verbose:
            print("step %2d: |f|: %12.6g mu: %8.3g rho: %8.3g"%(k, np.linalg.norm(f), mu, rho))

    else:
        reason = "max iter reached"

    if verbose:
        print(reason)

    if not full_output:
        return p
    else:
        return p, J, f
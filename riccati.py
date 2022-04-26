import numpy as np
# import scipy.special 

def cheb(n):
    """
    Returns the (n+1)*(n+1) differentiation matrix D and (n+1) Chebyshev nodes
    x for the standard 1D interval [-1, 1]. The matrix multiplies a vector of
    function values at these nodes to give an approximation to the vector of
    derivative values. Nodes are output in descending order from 1 to -1. 

    Parameters:

    Returns:
    """
    if n == 0:
        x = 1
        D = 0
    else:
        a = np.linspace(0.0, np.pi, n+1)
        x = np.cos(a)
        b = np.ones_like(x)
        b[0] = 2
        b[-1] = 2
        d = np.ones_like(b)
        d[1::2] = -1
        c = b*d
        X = np.outer(x, np.ones(n+1))
        dX = X - X.T
        D = np.outer(c, 1/c) / (dX + np.identity(n+1))
        D = D - np.diag(D.sum(axis=1))
    return D, x

def choose_stepsize(w, x0, h, epsh = 1e-14, p = 32):
    """
    Chooses the stepsize h over which the functions w(x), g(x) can be
    represented sufficiently accurately. p/2 nodes are randomly chosen over the
    interval [x0, x0+h] where p is the number of Chebyshev nodes osc_step()
    uses, i.e. the number of nodes that will be used to compute the Barnett
    series. The interpolated w, g are then checked at these points and compared
    to the actual function values. If the largest relative error in w, g
    exceeds epsh, h is halved.
    TODO: Actually add g, so far only have w
    """
    a = np.linspace(np.pi/(2*p), np.pi*(1 - 1/(2*p)), p)
    t = x0 + h/2 + h/2*np.cos(a)
    s = x0 + h/2 + h/2*cheb(p)[1]
    V = np.ones((p+1, p+1))
    R = np.ones((p, p+1))
#    print("sources: ", s)
#    print("Targets: ", t)
    for j in range(1, p+1):
        V[:, j] = V[:, j-1]*s
        R[:, j] = R[:, j-1]*t
    L = np.linalg.solve(V.T, R.T).T
    wana = w(t)
    west = np.matmul(L, w(s))
    maxwerr = max(np.abs((west - wana)/west))
    if maxwerr > epsh:
        print("Stepsize h = {} is too large with max error {}".format(h, maxwerr))
        return choose_stepsize(w, x0, 0.7*h, epsh = epsh, p = p)
    else:
        print("Chose stepsize h = {}".format(h))
        return h
    #TODO: what if h is too small to begin with?

def osc_step(w, x0, h, y0, dy0, epsres = 1e-12, n = 32):
    """
    Advances the solution from x0 to x0+h, starting from the initial conditions
    y(x0) = y0, y'(x0) = dy0. It uses the Barnett series of order o (up to and
    including the (o)th correction), and the underlying functions w, g are
    represented on an n-node Chebyshev grid.

    """
    success = True
    ddy0 = -w(x0)**2*y0
    D, x = cheb(n)
    ws = w(h/2*x + x0 + h/2)
    w2 = ws**2
    y = 1j*ws
    R = lambda y: 2/h*np.matmul(D, y) + y**2 + w2
    Ry = 0
    maxerr = 10*epsres
    prev_err = np.inf
    o = 0 # Keep track of number of terms
    while maxerr > epsres:
        o += 1
        y = y - Ry/(2*y)
        Ry = R(y)       
        maxerr = max(np.abs(Ry))
        if maxerr >= prev_err:
            print("Barnett series diverged after {} terms".format(o-1))
            success = False
            #TODO: Actually fail here
            break
        prev_err = maxerr
        #print("At iteration {}, max residual is Rx={}".format(o, maxerr))
    if success:
        print("Converged after {} terms".format(o))
    print("Residue = {}".format(maxerr))
    du1 = y
    du2 = np.conj(du1)
    u1 = h/2*np.linalg.solve(D, du1)
    u1 -= u1[-1]
    u2 = np.conj(u1)
    f1 = np.exp(u1)
    f2 = np.conj(f1)
    C = np.array([[1, 1], [du1[-1], du2[-1]]])
    ap, am = np.linalg.solve(C, np.array([y0, dy0]))
    y1 = ap*f1 + am*f2
    dy1 = ap*du1*f1 + am*du2*f2
    phase = np.imag(u1[0])
    return y1[0], dy1[0], maxerr, success, phase

def nonosc_step(x0, h, y0, dy0, o):
    """
    May need n as arg
    """
    pass

def solve(w, g, xi, xf, yi, dyi, eps = 1e-12, epsh = 1e-13, xeval = []):
    """
    Solves y'' + 2gy' + w^2y = 0 on the interval (xi, xf), starting from the
    initial conditions y(xi) = yi, y'(xi) = dyi. Keeps the residual of the ODE
    below eps, and returns an interpolated solution (dense output) at points
    xeval.

    Parameters:

    Returns:
    """
    # TODO: backwards integration
    xs = [xi]
    ys = [yi]
    dys = [dyi]
    phases = []
    successes = [True]
    y = yi
    dy = dyi
    n = 16 # How many points we use during Cheby interp
    p = n # How many points we use to choose h
    D, x = cheb(n)
    wi = w(xi)
    dwi = 2*np.matmul(D, w(xi + 1/2 + 1/2*x))[-1] 
    hi = wi/dwi
    print("Initial step: ", hi)
    h = choose_stepsize(w, xi, hi, epsh = epsh, p = p)
    xcurrent = xi
    while xcurrent < xf:
        print("x = {}, h = {}".format(xcurrent, h))
        # Attempt osc step of size h (for now always successful)
        y, dy, res, success, phase = osc_step(w, xcurrent, h, y, dy, epsres = eps, n = n)
        # Log step
        ys.append(y)
        dys.append(dy)
        xs.append(xcurrent + h)
        phases.append(phase)
        successes.append(success)
        # Advance independent variable and choose next step
        wnext = w(xcurrent + h)
        dwnext = 2/h*np.matmul(D, w(xcurrent + h/2 + h/2*x))[0]
        hnext = wnext/dwnext
#        hnext = wnext**3
        xcurrent += h
        h = choose_stepsize(w, xcurrent, hnext, epsh = epsh)
    return xs, ys, dys, successes, phases









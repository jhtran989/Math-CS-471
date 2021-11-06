from scipy import *
from poisson import poisson

def check_matvec(fpp_numeric, n, fcn):
    '''
    This is a testing harness to check your matrix-vector multiply
    This should print 0 
    '''
    fpp_numeric = fpp_numeric.reshape(-1,)

    Acheck = poisson((n, n), format='csr')
    h = 1./(n-1)
    Acheck *= 1/h**2
    pts = linspace(0,1,n)
    X,Y = meshgrid(pts, pts)
    X = X.reshape(-1,)
    Y = Y.reshape(-1,) 
    f_vals = fcn(X,Y)
    fpp_check = Acheck*f_vals

    #FIXME
    # print(f"shape fpp_check: {fpp_check.shape}")
    # print(f"shape fpp_numeric: {fpp_numeric.shape}")

    diff = fpp_check - fpp_numeric
    diff_norm = sqrt(dot(diff, diff)) / sqrt(dot(fpp_check, fpp_check))
    print("Matvec check (should be zero): " + str( diff_norm )) 


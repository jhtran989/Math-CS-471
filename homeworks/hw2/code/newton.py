from scipy import *
import numpy as np
def newton(f, x0, maxiter, tol):
    '''
    Newton's method of root finding

    input
    -----
    f           : function to find zero of
                  Interface for f:  f(x, 0)  returns f(x)
                                    f(x, 1)  returns f'(x)
    x0          : initial guess
    <maxiter>   :
    <tol>       :

    output
    ------
    data   : array of approximate roots
    '''


    maxiter = maxiter+1     # delete this line after you add a maxiter parameter
    data = zeros((maxiter,1))
    lin = zeros((maxiter,1))
    quad = zeros((maxiter,1))
    x = x0
    j = 0
    k = 0
    for i in range(maxiter):
        xold = x
        x = x - f(x, 0)/f(x, 1)

        s = "Iter  " + str(i) + "   Approx. root  " + str(x)
        print(s)

        data[i,0] = x

        if (i != 0):

            if(i > 1):
                err1 = np.abs(data[i-1,0] -data[i-2,0])
                err2 = np.abs(data[i,0] - data[i-1,0])
                if data[i-1,0] != 0:
                    lin[j,0] = np.abs(err2/err1)
                    j = j+1

                if data[i-1,0] != 0:
                    quad[k,0] = np.abs(err2/(err1**2))
                    k = k + 1
                    #print(quad)



            #np.savetxt('linear.tex', data, fmt='%.5e', delimiter='  &  ', newline=' \\\\\n')
            #print(lin)
        if (i != 0 and ((data[i,0] - data[i-1,0]) < tol or i == maxiter-1)):
            break
    ##
    # end for-loop

    return data[0:(i+1)], lin, quad


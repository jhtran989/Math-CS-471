from scipy import *
import numpy as np
from newton import newton

def fx(x, d):
    if d ==0:
        return x
    elif d == 1:
        return 1

def ftrig(x, d):
    if d == 0:
        return np.sin(x) + np.cos(x**2)
    elif d==1:
        return np.cos(x) - 2*x*np.sin(x**2)

def f1(x, d):
    if d == 0:
        # return f(x) = x^2
        return x**2
    elif d == 1:
        # return the derivative of f(x) = x^2
        return 2*x

fcn_list = [f1, fx, ftrig]

x0 = -0.5
for fcn in fcn_list:
    newt = newton(fcn, x0, 50, 1*10**(-10))
    data = newt[0]
    lin = newt[1]
    quad = newt[2]
    tests = np.arange(lin.size)
    flat = tests.reshape((lin.size,1))
    lin = np.column_stack((flat,lin))
    quad = np.column_stack((flat, quad))
    print(lin)
    print(quad)
    print()
    if fcn == f1:
        str_lin = 'x_2' + '_lin_test.txt'
        str_quad = 'x_2' + '_quad_test.txt'
    if fcn == fx:
        str_lin = 'x' + '_lin_test.txt'
        str_quad = 'x' + '_quad_test.txt'
    if fcn == ftrig:
        str_lin = 'trig' + '_lin_test.txt'
        str_quad = 'trig' + '_quad_test.txt'
    
    np.savetxt(str_lin, lin, fmt='%.5e', delimiter='  &  ', newline=' \\\\\\hline\n')
    np.savetxt(str_quad, quad, fmt='%.5e', delimiter='  &  ', newline=' \\\\\\hline\n')


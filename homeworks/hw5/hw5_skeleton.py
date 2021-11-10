from numpy import *
from matplotlib import pyplot
import matplotlib.animation as manimation
import os, sys

# HW5 Skeleton 

def RK4(f, y, t, dt, food_flag, alpha, gamma_1, gamma_2, kappa, rho, delta):
    '''food_flag, alpha, gamma_1, gamma_2, kappa, rho, delta)
    Carry out a step of RK4 from time t using dt
    
    Input
    -----
    f:  Right hand side value function (this is the RHS function)
    y:  state vector
    t:  current time
    dt: time step size
    
    food_flag:  0 or 1, depending on the food location
    alpha:      Parameter from homework PDF
    gamma_1:    Parameter from homework PDF
    gamma_2:    Parameter from homework PDF
    kappa:      Parameter from homework PDF
    rho:        Parameter from homework PDF
    delta:      Parameter from homework PDF
    

    Output
    ------
    Return updated vector y, according to RK4 formula
    '''
    print(y[0])
    h = dt
    # call RHS # y = RHS
    # Task: Fill in the RK4 formula
    for i in range(1, y.shape[0]):
        #print(y[i-1])
        # Task: insert formula for RK4
        ti = i*h
        print(y[i-1])
        k1 = f(y[i-1], ti, food_flag, alpha, gamma_1, gamma_2, kappa, rho, delta)
        print(y[i-1])
        k2 = f(ti + (h/2), y[i-1] + (h/2)*k1, food_flag, alpha, gamma_1, gamma_2, kappa, rho, delta)
        k3 = f(ti + (h/2), y[i-1] + (h/2)*k2, food_flag, alpha, gamma_1, gamma_2, kappa, rho, delta)
        k4 = f(ti + h, y[i-1] + h*k3, food_flag, alpha, gamma_1, gamma_2, kappa, rho, delta)

        y[i] = y[i-1] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)


    return y


def RHS(y, t, food_flag, alpha, gamma_1, gamma_2, kappa, rho, delta):
    '''
    Define the right hand side of the ODE

    '''
    print(y)
    N = y.shape[0]
    f = zeros_like(y)
    # y is B(t)
    # C(t) defined in pdf
    if food_flag == 0:
        C = array([0,0])
    if food_flag == 1:
        C = array([sin(alpha*t), cos(alpha*t)])

    F_food = gamma_1 * (C - y)
    F_follow = gamma_2 * (y[1] - y) # check
    F_flock = kappa * (avg(y) - y)
    # Task:  Fill this in by assigning values to f
    # f = n*2 vector sum of all components
    f = F_food + F_follow + F_flock
    return f


##
# Set up problem domain
t0 = 0.0        # start time
T = 10.0        # end time
nsteps = 50     # number of time steps

# Task:  Experiment with N, number of birds
N = 30

# Task:  Experiment with the problem parameters, and understand how each parameter affects the system
dt = (T - t0) / (nsteps-1.0)
gamma_1 = 2.0
gamma_2 = 8.0
alpha = 0.4
kappa = 4.0
rho = 2.0
delta = 0.5
food_flag = 1   # food_flag == 0: C(x,y) = (0.0, 0.0)
                # food_flag == 1: C(x,y) = (sin(alpha*t), cos(alpha*t))

# Intialize problem
y = random.rand(N,2)  # This is the state vector of each Bird's position.  The k-th bird's position is (y[k,0], y[k,1])
flock_diam = zeros((nsteps,))

RK4(RHS, y, (T-t0)/2, dt, food_flag, alpha, gamma_1, gamma_2, kappa, rho, delta)
# Initialize the Movie Writer
# --> The movie writing code has been done for you
#FFMpegWriter = manimation.writers['ffmpeg']
#writer = FFMpegWriter(fps=6)
#fig = pyplot.figure(0)
#pp, = pyplot.plot([],[], 'k+')
#rr, = pyplot.plot([],[], 'r+')
#pyplot.xlabel(r'$X$', fontsize='large')
#pyplot.ylabel(r'$Y$', fontsize='large')
#pyplot.xlim(-3,3)       # you may need to adjust this, if your birds fly outside of this box!
#pyplot.ylim(-3,3)       # you may need to adjust this, if your birds fly outside of this box!


# Begin writing movie frames
#with writer.saving(fig, "movie.mp4", dpi=1000):

    # First frame
#     pp.set_data(y[1:,0], y[1:,1])
#     rr.set_data(y[0,0], y[0,1])
#     writer.grab_frame()
#
#     t = t0
#     for step in range(nsteps):
#
#         # Task: Fill in the code for the next two lines
#         y = RK4(...)
#         flock_diam[step] = ...
#         t += dt
#
#         # Movie frame
#         pp.set_data(y[:,0], y[:,1])
#         rr.set_data(y[0,0], y[0,1])
#         writer.grab_frame()
#
#
# # Task: Plot flock diameter
# plot(..., flock_diam, ...)
#

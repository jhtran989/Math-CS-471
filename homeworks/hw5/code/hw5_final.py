from numpy import *
from matplotlib import pyplot
import matplotlib.animation as manimation
import os, sys
# import ffmpeg

# HW5 Skeleton

DEBUG = False
DIAMETER_DEBUG = False
SMELLY_BIRD = True  # we set it as the second bird (index 1)
PREDATOR = True

LEADER_INDEX = 0
SMELLY_BIRD_INDEX = 1

kappa_smelly = 4.0
rho_smelly = 3.0

# Predator
predator_location = random.rand(1, 2)
print(f"size of predator location: {predator_location.shape}")
print(f"predator location: {predator_location}")
rho_predator = 4.0

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

    if DEBUG:
        print(f"--------------------------------")
        print(f"t value: {t}")
        print(f"--------------------------------")

    # print(y[0])
    h = dt
    # call RHS # y = RHS
    # Task: Fill in the RK4 formula
    for i in range(1, y.shape[0] + 1):
        # print(y[i-1])
        # Task: insert formula for RK4

        #ti = i * h

        # print(y[i-1])
        k1 = f(y[i - 1], t, food_flag, alpha, gamma_1, gamma_2, kappa, rho,
               delta, y, i - 1)
        # print("k1", k1)
        k2 = f(y[i - 1] + (h / 2) * k1, t + (h / 2), food_flag, alpha, gamma_1,
               gamma_2, kappa, rho, delta, y, i - 1)
        # print("k2")
        k3 = f(y[i - 1] + (h / 2) * k2, t + (h / 2), food_flag, alpha, gamma_1,
               gamma_2, kappa, rho, delta, y, i - 1)
        k4 = f(y[i - 1] + h * k3, t + h, food_flag, alpha, gamma_1, gamma_2,
               kappa, rho, delta, y, i - 1)

        #FIXME: got lazy trying to switch the indices back so left it as
        # i - 1 for the current index
        # the new value will be stored in the same array (SAME index)
        y[i - 1] = y[i - 1] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    # print(y)

    return y


def RHS(y, t, food_flag, alpha, gamma_1, gamma_2, kappa, rho, delta, y_total,
        y_index):
    '''
    Define the right hand side of the ODE
    :param y_index:

    '''
    # print(y)
    N = y.shape[0]
    f = zeros_like(y)
    # y is B(t)
    # C(t) defined in pdf
    if food_flag == 0:
        C = array([0, 0])
    if food_flag == 1:
        C = array([sin(alpha * t), cos(alpha * t)])

    F_follow = gamma_2 * (y_total[0] - y)

    #print("y0", y_total[0])
    #
    # Task:  Fill this in by assigning values to f
    # f = n*2 vector sum of all components

    #FIXME: certain forces only apply to leader and others to the flock

    if DEBUG:
        print(f"bird index: {y_index}")

    f = 0

    if y_index == 0:
        F_food = gamma_1 * (C - y)
        f += F_food

        if DEBUG:
            print(f"F_food: {F_food}")

    f += F_follow
    if DEBUG:
        print(f"F_follow: {F_follow}")

    if y_index != 0:
        #FIXME: used the mean incorrectly, need to specify axis
        F_flock = kappa * (mean(y_total, axis=0) - y)

        distances = zeros((len(y_total), 2))
        current = y

        for n, item in enumerate(y_total):
            dist = sqrt(
                (item[0] - current[0]) ** 2 + (item[1] - current[1]) ** 2)
            distances[n, 1] = dist
            distances[n, 0] = n

        sorted = distances[argsort(distances[:, 1])]
        # print(sorted)
        neighbors = sorted[1:6]
        neighbors_coords = y_total[(neighbors[:, 0].astype(int))]
        # print(neighbors[:,0].astype(int))
        # print(neighbors_coords)
        # print(y_total)

        #TODO: need to fix F_repel -- maybe the sum only gives a scalar...
        numer = y - neighbors_coords
        F_repel = rho * sum(((numer) / (numer ** 2 + delta)), axis=0)

        f += F_flock + F_repel

        if DEBUG:
            print(f"F_flock: {F_flock}")
            print(f"numer: {numer}")
            print(f"F_repel: {F_repel}")

    if SMELLY_BIRD:
        if y_index == SMELLY_BIRD_INDEX:
            F_center_smelly = kappa_smelly * (mean(y_total, axis=0) - y)

            #F_smelly = gamma_1 * (C - y)
            f += F_center_smelly
        else:
            # similar to the repelling force above, we apply the repelling of
            # the 5 closest neighbors...
            numer = y - y_total[SMELLY_BIRD_INDEX]
            F_repel_smelly = rho_smelly * sum(((numer) / (numer ** 2 +
                                                          delta)), axis=0)
            f += F_repel_smelly

    if PREDATOR:
        numer = y - predator_location
        F_repel_predator = rho_predator * sum(((numer) / (numer ** 2 +
                                                      delta)), axis=0)
        f += F_repel_predator

    return f

if __name__ == "__main__":
    ##
    # Set up problem domain
    t0 = 0.0  # start time
    T = 10.0  # end time
    t_mid = (t0 + T) / 2 # mid time
    nsteps = 50  # number of time steps

    # Task:  Experiment with N, number of birds
    N = 10  # 10, 30, 100

    # Task:  Experiment with the problem parameters, and understand how each parameter affects the system
    dt = (T - t0) / (nsteps - 1.0)
    gamma_1 = 2.0  # 2.0
    gamma_2 = 8.0  # 8.0
    alpha = 0.4  # 0.4
    kappa = 4.0  # 4.0
    rho = 2.0  # 2.0
    delta = 0.5
    food_flag = 1  # food_flag == 0: C(x,y) = (0.0, 0.0)
    # food_flag == 1: C(x,y) = (sin(alpha*t), cos(alpha*t))

    # Intialize problem
    y = random.rand(N, 2)  # This is the state vector of each Bird's
    # position.  The k-th bird's position is (y[k,0], y[k,1])
    flock_diam = zeros((nsteps,))

    # RK4(RHS, y, (T - t0) / 2, dt, food_flag, alpha, gamma_1, gamma_2, kappa, rho,
    #     delta)

    # Initialize the Movie Writer
    # --> The movie writing code has been done for you
    FFMpegWriter = manimation.writers['ffmpeg']
    writer = FFMpegWriter(fps=6)
    fig = pyplot.figure(0)
    pp, = pyplot.plot([], [], 'k+')
    rr, = pyplot.plot([], [], 'r+')

    if SMELLY_BIRD:
        smelly, = pyplot.plot([], [], 'y+')

    if PREDATOR:
        predator, = pyplot.plot([], [], 'b+')

    pyplot.xlabel(r'$X$', fontsize='large')
    pyplot.ylabel(r'$Y$', fontsize='large')
    pyplot.xlim(-3,
                3)  # you may need to adjust this, if your birds fly outside of this box!
    pyplot.ylim(-3,
                3)  # you may need to adjust this, if your birds fly outside of this box!

    smelly_filename_addition = ""

    if SMELLY_BIRD:
        smelly_filename_addition = "_smelly"

    if PREDATOR:
        predator_filename_addition = "_predator"

    # Begin writing movie frames
    with writer.saving(fig, f"movie_{N}{smelly_filename_addition}"
                            f"{predator_filename_addition}.mp4",
                       dpi=1000):
        # First frame
        rr.set_data(y[0, 0], y[0, 1])
        if not SMELLY_BIRD:
            pp.set_data(y[1:, 0], y[1:, 1])
        else:
            smelly.set_data(y[SMELLY_BIRD_INDEX, 0], y[SMELLY_BIRD_INDEX, 1])
            pp.set_data(y[2:, 0], y[2:, 1])

        if PREDATOR:
            predator.set_data(predator_location[0][0], predator_location[0][1])

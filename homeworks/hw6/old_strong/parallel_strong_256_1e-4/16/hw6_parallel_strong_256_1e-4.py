import numpy
from numpy import *
from matplotlib import pyplot
from poisson import poisson
# speye generates a sparse identity matrix
from scipy.sparse import eye as speye
from scipy.sparse.linalg import splu

from mpi4py import MPI
import os
import sys

from time import time

global_maxiter = 400  # 250 go through code and refactor
global_tol = 1e-4  # 1e-10 1e-15 -- takes way to long for the strong scaling...

# Strong scaling -- repeat each 5 times and take the smallest of the 5 times
ntimings = 1  # changed from 5...each timing is too long for some reason
# hopefully with tol = 1e-6, the timings shouldn't take too long...
# actually, should still test with 1 timing for now...

# MPI Stuff
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

nprocs = comm.size

# print tol
if rank == 0:
    sys.stderr.write(f"tol: {global_tol}\n")

# Plots Stuff
parallel_root = f"parallel/"
# parallel_plots_dir = f"{parallel_root}plots/"
# os.makedirs(parallel_plots_dir, exist_ok=True)

if rank == 0:  # reduce any clashes when multiples processes try to make the
    # directory...
    os.makedirs(parallel_root, exist_ok=True)

# move to main below
# os.makedirs(parallel_plots_dir, exist_ok=True)

# DEBUG Stuff
# print convergence check
CONVERGENCE_CHECK = True
FIRST_TIME_STEP_CHECK = False
LAST_TIME_STEP_CHECK = False
LAST_JACOBI_ITERATION_CHECK = False
CONVERGENCE_DEBUG = False
NORM_DEBUG = False
MATVEC_CHECK = False
JACOBI_DEBUG = False
INITIALIZATION_DEBUG = False
FINAL_DEBUG = False

# plot individual time steps -- still create i = 0 plot for reference
PLOT_TIME_STEP = False

# added the communication under condition as well to reduce
# total runtime
# actually, we could just use the norm function above...with
# allGather
ERROR_NORM = True

'''
    # Problem Preliminary: MPI cheat sheet

    # For the later parallel part, you'll need to use MPI. 
    # Here are the most useful commands. 

    # Import MPI at start of program
    from mpi4py import MPI

    # Initialize MPI
    comm = MPI.COMM_WORLD

    # Get your MPI rank
    rank = comm.Get_rank()

    # Send "data" (an array of doubles) to rank 1 from rank 0
    comm.Send([data, MPI.DOUBLE], dest=1, tag=77)

    # Receive "data" (the array of doubles) from rank 0 (on rank 1)
    comm.Recv([data, MPI.DOUBLE], source=0, tag=77)

    # Carry out an all-reduce, to sum over values collected from all processors
    # Note: "part_norm" and "global_norm" are length (1,) arrays.  The result
    #        of the global sum will reside in global_norm.
    # Note: If MPI.SUM is changed, then the reduce can multiple, subtract, etc.
    comm.Allreduce(part_norm, global_norm, op=MPI.SUM)

    # For instance, a simple Allreduce program is the following.
    # The result in global_norm will be the total number of processors
    from scipy import *
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    part_norm = array([1.0])
    global_norm = zeros_like(part_norm) 

    comm.Allreduce(part_norm, global_norm, op=MPI.SUM)

    if (rank == 0):
        print(global_norm)
'''


##
# Problem Definition
# Task: figure out (1) the exact solution, (2) f(t,x,y), and (3) g(t,x,y)
#
# Governing PDE:
# u_t = u_xx + u_yy + f,    on the unit box [0,1] x [0,1] and t in a user-defined interval
#
# with an exact solution of
# u(t,x,y) = ...
#
# which in turn, implies a forcing term of f, where
# f(t,x,y) = ...
#
# an initial condition of
# u(t=0,x,y) = ...
#
# and a boundary condition in space when (x,y) is on the boundary of
# g(t,x,y) = u(t,x,y) = ...
#
##

# Declare the problem
def uexact(t, x, y):
    # Task: fill in exact solution
    # return sin(pi*t)*sin(pi*x)*sin(pi*y) # this is the exact solution
    return cos(pi * t) * cos(pi * x) * cos(pi * y)
    # return (t-0.9)*(x**2)*(y**2)


def f(t, x, y):
    # Forcing term
    # This should equal u_t - u_xx - u_yy

    # Task: fill in forcing term
    # return pi*(cos(pi*t))*sin(pi*x)*sin(pi*y) + 2*pi*pi*uexact(t, x, y)
    # this is f, change for new cos function
    return pi * (-sin(pi * t)) * cos(pi * x) * cos(
        pi * y) + 2 * pi * pi * uexact(t, x, y)
    # return 1 - (((t - 0.9) * 2 * (y ** 2)) + ((t - 0.9) * 2 * (x ** 2)))


# try commenting out and using sin sin sin because it has zero boundary conditions
# issue might be with x in backward euler
# issue in scaling
# check g, should be 0

##
# Task in serial: Implement Jacobi

# Task in parallel: Extend Jacobi to parallel (as described below), with a
#                   parallel matrix-vector product and parallel vector norm.
#                   It is suggested to write separate subroutines for the
#                   parallel norm and the parallel matrix-vector product, as
#                   this will make your code much, much easier to debug.
#
#                   For instance, the instructor wrote a routine
#                   "matrix_vector()" which computes A*x with an interface
#                          matrix_vector(A, x, N, comm)
#                   where A is the matrix, x is the vector to multiply,
#                   N is the number of domain rows (excluding boundary rows)
#                   (like 8 for a 8x8 grid), and comm is the MPI communicator.
##
def jacobi(A, b, x0, tol, maxiter, n_local, comm):
    '''
    Carry out the Jacobi method to invert A

    Input
    -----
    A <CSR matrix>  : Matrix to invert (G in slides)
    b <array>       : Right hand side
    x0 <array>      : Initial solution guess

    Output
    ------
    x <array>       : Solution to A x = b
    :param n_local:
    :param comm:
    '''

    if JACOBI_DEBUG:
        sys.stderr.write(f"rank: {rank}\n")
        sys.stderr.write(f"n_local: {n_local}\n")

    # This useful function returns an array containing diag(A)
    D = A.diagonal()
    # print(f"Diagonal: {D}")
    D_inv = 1 / D  # linalg.inv(D)
    # print(f"Diagonal inverse: {D_inv}")

    # compute initial residual norm
    # r0_global = array([0.0])
    # r0 = ravel(b - A*x0)
    # r0 = array([dot(r0, r0)])
    # comm.Allreduce(r0, r0_global, op=MPI.SUM)
    #
    # r0 = sqrt(r0_global[0])  # this is the init residual

    # IMPORTANT: need to communicate x initially since x0 still has the
    # "incorrect values" from the finish of the last time iteration...

    neighbor_communication(x0, n_local, comm)

    r0_vector = b - A * x0

    # in order to calculate the norm correctly, only use the LOCAL
    # PORTION of the residual -- INCLUDING r0
    if nprocs > 1:
        if rank == 0:
            r0_vector = r0_vector[:n_local]
        elif rank == (nprocs - 1):
            r0_vector = r0_vector[-n_local:]
        else:
            r0_vector = r0_vector[n_local:-n_local]

    r0_norm = norm(r0_vector, comm)

    if rank == 0:
        if CONVERGENCE_CHECK:
            if FIRST_TIME_STEP_CHECK or LAST_TIME_STEP_CHECK:
                sys.stderr.write(f"b: {b}\n")
                sys.stderr.write(f"x0: {x0}\n")
                sys.stderr.write(f"r0 vector: {r0_vector}\n")
                sys.stderr.write(f"r0 norm: {r0_norm}\n")

    # print(f"r0: {r0}")

    I = speye(A.shape[0], format='csr')
    # Start Jacobi iterations
    # Task in serial: implement Jacobi method and halting tolerance based on the residual norm

    # Task in parallel: extend the matrix-vector multiply to the parallel setting.
    #                   Additionally, you'll need to compute a norm of the residual in parallel.
    x = zeros((maxiter + 1, A.shape[0]))  # only need first and last
    x[0] = x0
    last_i = 0
    for i in range(maxiter):
        # << Jacobi algorithm goes here >>
        # Lecture 26, Slide 23

        # old
        # x[i+1] = (x[i] - (A*x[i])*D_inv) + D_inv*b
        x[i + 1] = jacobi_step(A, x[i], b, D_inv, n_local, comm)

        # rk_global = array([0.0])
        #
        # rk = b - A*x[i+1]
        # # print(f"before ravel:")
        # #print(f"rk: {rk}")
        # # print(f"rk shape: {rk.shape}")
        # rk = ravel(rk)  # does nothing...
        # rk = array([dot(rk, rk)])
        #
        # comm.Allreduce(rk, rk_global, op=MPI.SUM)
        #
        # rk = sqrt(rk_global[0])  # this is the init residual

        # IMPORTANT: need to communicate x AFTER THE UPDATE for x[i + 1] --
        # only updated LOCAL PORTION, not neighboring values of x...
        neighbor_communication(x[i + 1], n_local, comm)

        if CONVERGENCE_DEBUG:
            print(f"Jacobi iteration: {i}")

        rk_vector = b - A * x[i + 1]

        # in order to calculate the norm correctly, only use the LOCAL
        # PORTION of the residual
        if nprocs > 1:
            if rank == 0:
                rk_vector = rk_vector[:n_local]
            elif rank == (nprocs - 1):
                rk_vector = rk_vector[-n_local:]
            else:
                rk_vector = rk_vector[n_local:-n_local]

        rk_norm = norm(rk_vector, comm)

        # print(f"after ravel:")
        # print(f"rk: {rk}")
        # print(f"rk shape: {rk.shape}")
        # rk = sqrt(dot(rk, rk))
        # print(f"rk: {rk}")
        last_i = i

        # print(f"r0: {r0}")
        # print(f"rk: {rk}")
        if rk_norm / r0_norm <= tol:
            if rank == 0:
                if CONVERGENCE_CHECK:
                    if FIRST_TIME_STEP_CHECK:
                        #sys.stderr.write(f"b: {b}\n")
                        sys.stderr.write(f"n_local: {n_local}\n")
                        sys.stderr.write(f"rk_vector: {rk_vector}\n")
                        sys.stderr.write(f"rk norm: {rk_norm}\n")
                        sys.stderr.write(f"Residual ratio: "
                                         f"{rk_norm / r0_norm}\n")
                        sys.stderr.write(f"First time step\n")
                        sys.stderr.write(f"did converge, i = {i}\n")
                    elif LAST_TIME_STEP_CHECK:
                        #sys.stderr.write(f"b: {b}\n")
                        sys.stderr.write(f"n_local: {n_local}\n")
                        sys.stderr.write(f"rk_vector: {rk_vector}\n")
                        sys.stderr.write(f"rk norm: {rk_norm}\n")
                        sys.stderr.write(f"Residual ratio: "
                                         f"{rk_norm / r0_norm}\n")
                        sys.stderr.write(f"Last time step\n")
                        sys.stderr.write(f"did converge, i = {i}\n")

            break

    # Task: Print if Jacobi did not converge. In parallel, only rank 0 should print.
    if rk_norm / r0_norm > tol:
        if rank == 0:
            if CONVERGENCE_CHECK:
                if FIRST_TIME_STEP_CHECK:
                    #sys.stderr.write(f"b: {b}\n")
                    sys.stderr.write(f"n_local: {n_local}\n")
                    sys.stderr.write(f"rk_vector: {rk_vector}\n")
                    sys.stderr.write(f"rk norm: {rk_norm}\n")
                    sys.stderr.write(f"Residual ratio: "
                                     f"{rk_norm / r0_norm}\n")
                    sys.stderr.write(f"First time step\n")
                    sys.stderr.write(f"did NOT converge\n")
                elif LAST_TIME_STEP_CHECK:
                    #sys.stderr.write(f"b: {b}\n")
                    sys.stderr.write(f"n_local: {n_local}\n")
                    sys.stderr.write(f"rk_vector: {rk_vector}\n")
                    sys.stderr.write(f"rk norm: {rk_norm}\n")
                    sys.stderr.write(f"Residual ratio: "
                                     f"{rk_norm / r0_norm}\n")
                    sys.stderr.write(f"Last time step\n")
                    sys.stderr.write(f"did NOT converge\n")

    # print("x", x.shape)
    # return x[last_i + 1:last_i + 2]

    # return x[last_i+1:last_i+2] # this is good
    if rank == 0:
        return x[last_i][0:n_local]  # this is good
    elif rank == (nprocs - 1):
        return x[last_i][-n_local:]
    else:
        return x[last_i][n_local:-n_local]


def neighbor_communication(x, n_local, comm):
    # communication only with more than 1 processes...

    if JACOBI_DEBUG:
        if rank == 0:
            sys.stderr.write(f"during neighbor communication...\n")

        sys.stderr.write(f"rank: {rank}\n")
        sys.stderr.write(f"x: {x}\n")
        sys.stderr.write(f"x shape: {x.shape}\n")

    if nprocs > 1:
        # Send to top neighbor (if not rank 0)
        # rank - 1
        if rank != 0:
            if rank == (nprocs - 1):
                comm.Send([x[-n_local:], MPI.DOUBLE], dest=rank - 1, tag=77)
            else:
                comm.Send([x[n_local:-n_local], MPI.DOUBLE], dest=rank - 1,
                          tag=77)

        # Task: Send to bottom neighbor (if not last rank)
        # rank + 1
        if rank != (nprocs - 1):
            if rank == 0:
                comm.Send([x[:n_local], MPI.DOUBLE], dest=rank + 1, tag=77)
            else:
                comm.Send([x[n_local:-n_local], MPI.DOUBLE], dest=rank + 1,
                          tag=77)

        # REMEMBER TO RECEIVE in a DIFFERENT PART of x
        # Task: Receive from right neighbor (if not last rank)
        # Task: receive from the top neighbor
        # rank - 1
        if rank != 0:
            # if rank == (nprocs - 1):
            #     comm.Recv([x[:n_local], MPI.DOUBLE], source=rank - 1, tag=77)
            # else:
            #     comm.Recv([x[:n_local], MPI.DOUBLE], source=rank - 1,
            #               tag=77)

            comm.Recv([x[:n_local], MPI.DOUBLE], source=rank - 1, tag=77)

        # Task: receive from the bottom neighbor
        # rank + 1
        if rank != (nprocs - 1):
            # if rank == 0:
            #     comm.Recv([x[-n_local:], MPI.DOUBLE], source=rank + 1, tag=77)
            # else:
            #     comm.Recv([x[-n_local:], MPI.DOUBLE], source=rank + 1,
            #               tag=77)

            comm.Recv([x[-n_local:], MPI.DOUBLE], source=rank + 1, tag=77)


def matrix_vector(A, x, n_local, comm):
    """
    N is renamed to n_local

    :param A:
    :param x:
    :param n_local:
    :param comm:
    :return:
    """

    neighbor_communication(x, n_local, comm)

    # (I - A*D_inv)*x
    # A is G in this case
    # x - (A*x)*D_inv
    return A * x


def jacobi_step(A, x, b, D_inv, n_local, comm):
    """
    Changed the function definition/prototype from matrix_vector(A, x, N,
    comm) to compute the ENTIRE line, including b and D_inv and parameters

    :param A:
    :param x:
    :param b:
    :param D_inv:
    :param n_local:
    :param comm:
    :return:
    """
    # communicate the values of x first (the x from the halo regions have the
    # INCORRECT values in the local portion -- A gets cut off from the global)

    # assume rank 0 is the top-most row and numbering goes down

    # different lengths of x for the first and last processes (halo regions
    # different from the others)
    # --> Pay attention to the data-type

    if JACOBI_DEBUG:
        if rank == 0:
            sys.stderr.write(f"before neighbor communication:\n")

    neighbor_communication(x, n_local, comm)

    if JACOBI_DEBUG:
        if rank == 0:
            sys.stderr.write(f"after neighbor communication:\n")

    # (I - A*D_inv)*x
    # A is G in this case
    # x - (A*x)*D_inv
    return (x - (A * x) * D_inv) + D_inv * b


def norm(r_vector, comm):
    r_global_norm = array([0.0])
    r_local_dot = array([dot(r_vector, r_vector)])

    # communication with only multiple processes
    if nprocs > 1:
        comm.Allreduce(r_local_dot, r_global_norm, op=MPI.SUM)
    else:
        r_global_norm = r_local_dot

    r_global_norm_scalar = sqrt(r_global_norm[0])

    if rank == 0:
        if NORM_DEBUG:
            if FIRST_TIME_STEP_CHECK or LAST_TIME_STEP_CHECK:
                sys.stderr.write(f"global norm: {r_global_norm_scalar}\n")

    return r_global_norm_scalar


def euler_backward(A, u, ht, f, g, n_local_domain):
    '''
    Carry out backward Euler for one time step

    Input
    -----
    A <CSR matrix>  : Discretization matrix of Poisson operator
    u <array>       : Current solution vector at previous time step (u is ui)
    ht <scalar>     : Time step size
    f <array>       : Current forcing vector
    g <array>       : Current vector containing boundary condition information

    Output
    ------
    u at the next time step
    :param n_local_domain:

    '''
    # print("g", g)
    # Task: Form the system matrix for backward Euler
    I = speye(A.shape[0], format='csr')
    G = I - ht * A
    b = u + ht * g + ht * f  # G*u # PP 26 pg 22   fix this
    # Ainv.solve(eye(A.shape[0]) - ht*A)*(u+ht*f+ht*f)
    # Task: return solution from Jacobi, which takes a time-step forward in time by "ht"
    # jacobi(A, b, x0, tol, maxiter):
    return jacobi(G, b, u, global_tol, global_maxiter, n_local_domain, comm)
    # Ainv = splu(G)
    # return Ainv.solve(b)
    # return G_inv.solve() # exact solve from lab to check jacobi, if not fixed, than issue is not in jacobi


# Helper function provided by instructor for debugging.  See how matvec_check
# is used below.
def matvec_check(A, X, Y, N, comm, h):
    '''
    This function runs

       (h**2)*A*ones()

    which should yield an output that is zero for interior points,
    -1 for points next to a Wall, and -2 for the four corner points

    All the results are printed to the screen.

    Further, it is assumed that you have a function called "matrix_vector()"
    that conforms to the interface described above for the Jacobi routine.  It
    is assumed that the results of matrix_vector are only accurate for non-halo
    rows (similar to compute_fd).

    PERSONAL ADDITION:
    added an "h" parameter to scale the matrix-vector multiplication by (
    h**2) to cancel out the (1/h**2) factor initially applied to A...
    :param h:
    '''

    # Defined above (global variables)
    # nprocs = comm.size
    # my_rank = comm.Get_rank()

    o = ones((A.shape[0],))

    # created a separate matrix_vector_plain
    # wait...no communication is needed, silly instruction
    # also, A is already scaled by
    # oo = matrix_vector(A, o, n_local_original, comm)
    oo = (h ** 2) * A * o
    if rank != 0:
        oo = oo[N:]
        X = X[N:]
        Y = Y[N:]
    if rank != (nprocs - 1):
        oo = oo[:-N]
        X = X[:-N]
        Y = Y[:-N]

    # move import to the top of file
    # import sys
    for i in range(oo.shape[0]):
        sys.stderr.write(f"rank: {rank}\n")
        sys.stderr.write(
            "X,Y: (%1.2e, %1.2e),  Ouput: %1.2e\n" % (X[i], Y[i], oo[i]))


# def matrix_vector_plain(A, x, n_local, comm):
#


###########
# This code block chooses the final time and problem sizes (nt, n) to loop over.
# - You can automate the selection of problem sizes using if/else statements or
#   command line parameters.  Or, you can simply comment in (and comment out)
#   lines of code to select your problem sizes.
#
# - Note that N_values corresponds to the number of _interior_ (non-boundary) grid
#   points in one coordinate direction.  Your total number of grid points in one
#   coordinate direction would be (N_values[k] + 2).
#
# - The total number of spatial points is (N_values[k] + 2)^2

if __name__ == "__main__":
    # Use these problem sizes for your error convergence studies
    # Nt_values = array([8, 8*4, 8*4*4, 8*4*4*4])
    # N_values = array([8, 16, 32, 64 ])
    # T = 0.5

    # Changed for our special case (second function above)
    # Nt_values = array([12 * (4 ** i) for i in range(4)])  # 4
    # N_values = array([8 * (2 ** i) for i in range(4)])  # 4
    #
    # print(f"N time values: {Nt_values}")
    # print(f"N values: {N_values}")

    # Strong scaling study where the ratio ht/h**2 stays around 4 for all four
    # cases (maybe command line arguments?)
    # so, the number of processes to be used are 2, 4, 8, 16, 32, 64
    # scale T so that the ratio ht/h**2 stays around 4 for all four
    # cases
    Nt_values = array([1024])  # 1024
    N_values = array([256])  # 512
    T = 4.0 * (1 / (N_values[0] ** 2)) * Nt_values[0]  # 1/64

    # Nt_values = array([12 * (4 ** 3)])  # 8*4 -> 100
    # N_values = array([8 * (2 ** 3)])  # 16
    # T = 0.75  # 0.5

    # IMPORTANT: reduced the time quite a bit -- problem seems to be
    # calculating uexact for the ENTIRE grid on process 0 at EACH TIME STEP

    # Weak scaling debug -- compare timings...
    # # Nt_values = array([16 * (4 ** power)])  # 16 -- initial
    # # N_values = array([48 * (2 ** power)])  # 48 -- initial
    # # test -- debug
    # power = int(log(nprocs) / log(4))
    # Nt_values = array([12 * (4 ** power)])  # 16 -- initial
    # N_values = array([8 * (2 ** power)])  # 48 -- initial
    # T = 4.0 * (1 / (N_values[0] ** 2)) * Nt_values[0]  # 1/36

    # keep track of all the timings to find the min time at the end
    timings_array = zeros((ntimings, ))

    # Initial stuff (comm)
    if rank == 0:
        sys.stderr.write(f"number of processes: {nprocs}\n")
        sys.stderr.write(f"Nt values: {Nt_values}\n")
        sys.stderr.write(f"N values: {N_values}\n")
        sys.stderr.write(f"T: {T}\n")

    # One very small problem for debugging
    # Nt_values = array([8])  # 8*4 -> 100
    # N_values = array([8])  # 16
    # T = 0.5  # 0.5

    # Changed for our special case (second function above)
    # Nt_values = array([12]) # 8*4 -> 100
    # N_values = array([8])  # 16
    # T = 0.75  # 0.5

    # part_norm = array([1.0])
    # global_norm = zeros_like(part_norm)

    # send and receive reference
    # comm.Send([data, MPI.INT], dest=1, tag=77)
    # comm.Recv([data, MPI.INT], source=0, tag=77)

    # comm.Allreduce(part_norm, global_norm, op=MPI.SUM)

    # if (rank == 0):
    #     print(global_norm)

    # Parallel Task: Change T and the problem sizes for the weak and strong scaling studies
    #
    # For instance, for the strong scaling, you'll want
    # Nt_values = array([1024])
    # N_values = array([512])
    # T = ...
    #
    # And for the first weak scaling run, you'll want
    # Nt_values = array([16])
    # N_values = array([48])
    # T = 0.03

    ###########

    # enclose the usual algorithm in a ntimings loop and take the min time
    for timing_index in range(ntimings):
        # Define list to contain the discretization error from each problem size
        error = []

        # Begin loop over various numbers of time points (nt) and spatial grid sizes (n)
        for (nt, n) in zip(Nt_values, N_values):
            # create dir in plots for each value in Nt_values
            # parallel_plots_current_dir = f"{parallel_plots_dir}{nt}/"
            # os.makedirs(parallel_plots_current_dir, exist_ok=True)

            parallel_root_current = f"{parallel_root}nprocs={nprocs},n={n}," \
                                    f"nt={nt},T={T}/"
            parallel_plots_dir_current = f"{parallel_root_current}plots/"

            if rank == 0:  # again, reduce clashing when trying to create it
                os.makedirs(parallel_plots_dir_current, exist_ok=True)

            # Declare time step size
            t0 = 0.0
            ht = (T - t0) / float(nt - 1)

            # Declare spatial grid size.  Note that we divide by (n + 1) because we are
            # accounting for the boundary points, i.e., we really have n+2 total points
            h = 1.0 / (n + 1.0)

            # Task in parallel:
            # Compute which portion of the spatial domain the current MPI rank owns,
            # i.e., compute "start", "end", "start_halo", and "end_halo"
            #
            #  - This will be similar to HW4.  Again, assume that n/nprocs divides evenly
            #
            #  - Because of the way that we are handling boundary domain rows
            #    (remember, we actually have n+2 domain rows), you may want to
            #    shift start and end up by "+1" when compared to HW4
            #
            #  - Lastly, just like with HW4, Cast start and end as integers, e.g.,
            #    start = int(....)
            #    end = int(....)
            #    start_halo = int(....)
            #    end_halo = int(....)

            # Remember, we assume a Dirichlet boundary condition, which simplifies
            # things a bit.  Thus, we only want a spatial grid from
            # [h, 2h, ..., 1-h] x [h, 2h, ..., 1-h].
            # We know what the solution looks like on the boundary, and don't need to solve for it.
            #
            # Task: fill in the right commands for the spatial grid vector "pts"
            # Task in parallel: Adjust these computations so that you only compute the local grid
            #                   plus halo region.  Mimic HW4 here.
            n_local_original = n // nprocs

            start = rank * n_local_original
            end = (rank + 1) * n_local_original

            # implement halo regions
            if rank != 0:
                start_halo = start - n_local_original
            else:
                start_halo = start
            # start_halo = <start - 1, unless k == 0>
            if rank != (nprocs - 1):
                end_halo = end + n_local_original
            else:
                end_halo = end

            start_Y = h + start_halo * h
            end_Y = h + end_halo * h

            X_pts = linspace(h, 1 - h, n)

            # changed n_local_original to end_halo - start_halo
            Y_pts = linspace(start_Y, end_Y - h, end_halo - start_halo)  # -h is
            # IMPORTANT (n+2 points, but only n+1 SPACINGS)
            # for 1 process, end = h and end_Y = h + n * h = h * (n + 1)...

            X, Y = meshgrid(X_pts, Y_pts)

            X = X.reshape(-1, )
            Y = Y.reshape(-1, )

            if INITIALIZATION_DEBUG:
                print("pts", Y_pts)

            # keep track of the entire mesh grid for global calculations (uexact)
            # can't use uexact(t, X, X) since the values of X are ordered
            # differently than Y
            if rank == 0:
                X_global, Y_global = meshgrid(X_pts, X_pts)
                X_global = X_global.reshape(-1, )
                Y_global = Y_global.reshape(-1, )

            # Declare spatial discretization matrix
            # Task: what dimension should A be?  remember the spatial grid is from
            #       [h, 2h, ..., 1-h] x [h, 2h, ..., 1-h]
            #       Pass in the right size to poisson.
            # Task in parallel: Adjust the size of A, that is A will be just a processor's
            #                   local part of A, similar to HW4
            # in HW4 poisson((end_halo - start_halo, n), format='csr')

            sizex = n
            sizey = end_halo - start_halo

            A = poisson((sizey, sizex), format='csr')

            # Task: scale A by the grid size
            A = (1 / h ** 2) * (A)

            # Declare initial condition
            #   This initial condition obeys the boundary condition.
            # print("X.shape", X.shape)
            # print("Y.shape", Y.shape)
            u0_local = uexact(0, X, Y)

            # Declare storage
            # Task: Declare "u" and "ue".  What sizes should they be?  "u" will store the
            #       numerical solution, and "ue" will store the exact solution.
            # Task in parallel: Adjust the sizes to be only for this processor's
            #                   local portion of the domain.
            # print("maxiter", maxiter)
            # print("A.size[0]", A.shape[0])

            A_shape = int((A.shape[0]))

            if INITIALIZATION_DEBUG:
                sys.stderr.write(f"rank: {rank}\n")
                sys.stderr.write(f"A_shape: {A_shape}\n")

            # size of u_local should be A_shape for each vector
            u_local = zeros((nt, A_shape))
            ue_local = zeros((nt, A_shape))

            # change index for columns (current values in middle, depending on
            # where it is)
            # if rank == 0:
            #     u[0, 0:n_local] = u0_local  # u0[-1:]  # starts at exact solution
            #     ue[0, 0:n_local] = u0_local  # [-1:]
            # elif rank == (nprocs - 1):
            #     u[0, :] = u0_local  # u0[-1:]  # starts at exact solution
            #     ue[0, :] = u0_local  # [-1:]
            # else:
            #     u[0, :] = u0_local  # u0[-1:]  # starts at exact solution
            #     ue[0, :] = u0_local  # [-1:]

            # n_local_domain should be the number of domain rows for each
            # processor (excluding the halo regions...)
            # n_local_domain = A_shape

            if nprocs == 1:
                n_local_domain = A_shape
            else:
                if rank == 0 or rank == (nprocs - 1):
                    n_local_domain = A_shape // 2
                else:
                    n_local_domain = A_shape // 3

            if rank == 0:
                start_index = 0
                end_index = n_local_domain
            elif rank == (nprocs - 1):
                start_index = -n_local_domain
                end_index = A_shape
            else:
                start_index = n_local_domain
                end_index = -n_local_domain

            u_local[0][start_index:end_index] = u0_local[start_index:end_index]
            # u0[-1:]  # starts at exact solution
            ue_local[0][start_index:end_index] = u0_local[start_index:end_index]
            # [-1:]

            # matvec check after initialization of A and number of domain rows
            # (n_local_domain)
            # should only be used for small problem sizes (small values of n)
            if MATVEC_CHECK:
                matvec_check(A, X, Y, n_local_domain, comm, h)

            # section of u each process will get
            # u_local_size = A.shape[0]
            #
            # start_index = rank * u_local_size
            # end_index = (rank + 1) * u_local_size
            #
            # u_local_0 = u0_local[start_index:end_index]

            # # Send to left neighbor (if not rank 0)
            # # --> Pay attention to the data-type
            # if rank != 0:
            #     comm.Send([x_local[0:1], MPI.DOUBLE], dest=rank - 1, tag=77)
            #
            # # Task: Send to right neighbor (if not last rank)
            # if rank != (num_ranks - 1):
            #     comm.Send([x_local[-1:], MPI.DOUBLE], dest=rank + 1, tag=77)
            #
            # # Task: Receive from right neighbor (if not last rank)
            # if rank != (num_ranks - 1):
            #     comm.Recv([x_right[0:1], MPI.DOUBLE], source=rank + 1, tag=77)
            #
            # # Task: Receive from left neighbor (if not rank 0)
            # if rank != 0:
            #     comm.Recv([x_left[0:1], MPI.DOUBLE], source=rank - 1, tag=77)

            # Set initial condition
            if INITIALIZATION_DEBUG:
                print("u", u_local.shape)
                print("u0", u0_local)

            start_domain_index = rank * n_local_domain
            end_domain_index = (rank + 1) * n_local_domain

            # set global variables only on rank 0 (for plotting at the end)
            # same indexing with global and local ONLY on rank 0
            if rank == 0:
                u_global = zeros((nt, n ** 2))
                u_global[0][start_domain_index:end_domain_index] = \
                    u0_local[start_domain_index:end_domain_index]

                ue_global = zeros((nt, n ** 2))
                ue_global[0][start_domain_index:end_domain_index] = \
                    u0_local[start_domain_index:end_domain_index]

            # communication only with more than 1 processes...
            # if nprocs > 1:
            #     if rank != 0:
            #         if rank == (nprocs - 1):
            #             comm.Send([u0_local[-n_local_domain:], MPI.DOUBLE],
            #                       dest=0,
            #                       tag=77)
            #         else:
            #             comm.Send(
            #                 [u0_local[n_local_domain:-n_local_domain], MPI.DOUBLE],
            #                 dest=0,
            #                 tag=77)
            #
            #     if rank == 0:
            #         for rank_num in range(1, nprocs):
            #             start_index_inner = rank_num * n_local_domain
            #             end_index_inner = (rank_num + 1) * n_local_domain
            #
            #             comm.Recv([u_global[0]
            #                        [start_index_inner:end_index_inner], MPI.DOUBLE],
            #                       source=rank_num, tag=77)

            if PLOT_TIME_STEP:
                if rank == 0:
                    pyplot.figure(0)
                    pyplot.imshow(u_global[0, :].reshape(n, n), origin='lower',
                                  extent=(0, 1, 0, 1))
                    pyplot.colorbar()
                    pyplot.xlabel('X')
                    pyplot.ylabel('Y')
                    pyplot.title(f"Solution, i={0}")
                    pyplot.savefig(f"{parallel_plots_dir_current}solution_{0}.png")
                    pyplot.close(pyplot.figure(0))

            # Testing harness for parallel part: Only comment-in and run for the smallest
            # problem size of 8 time points and an 8x8 grid
            #   - Assumes specific structure to your mat-vec multiply routine
            #    (described above in comments)
            # matvec_check( (h**2)*A, X, Y, n-2, comm)

            TIME_DEBUG = False
            # t = t0

            # Timings are done only on the calculations part (with backwards
            # euler) for ALL time steps -- time is also kept only on process 0
            # since process 0 also receives the other values of x to plot each
            # time step out...

            # start time
            if rank == 0:
                start_time = time()

            # Run time-stepping over "nt" time points
            for i in range(1, nt):
                # set TIME_CHECK above to print the number of iterations of
                # Jacobi for checking
                if i == 1:
                    FIRST_TIME_STEP_CHECK = True
                elif i == (nt - 1):
                    LAST_TIME_STEP_CHECK = True
                else:
                    # only need to reset FIRST_TIME_STEP_CHECK
                    if FIRST_TIME_STEP_CHECK:
                        FIRST_TIME_STEP_CHECK = False

                t = t0 + i * ht
                # ERROR - DON'T use t0 (only initial iteration)

                if TIME_DEBUG:
                    print(f"------------------------------")
                    print("t", t)
                    print("i (time)", i)

                # Task: We need to store the exact solution so that we can compute the error
                # print("ue.shape", ue.shape)
                # print("uexact.shape", uexact(t, X, Y).shape)

                # print(f"rank: {rank}")
                # print(f"A shape: {A_shape}")
                # print(f"ue_local shape: {ue_local.shape}")
                # print(f"uexact shape: {uexact(t, X, Y).shape}")
                # print(f"X shape: {X.shape}")
                # print(f"Y shape: {Y.shape}")

                ue_local[i, :] = uexact(t, X, Y)

                # Task: Compute boundary contributions for the current time value of i*ht
                #       Different from HW4, need to account for numeric error, hence "1e-12" and not "0"
                g = zeros((A.shape[0],))
                # print("a_shape[0]", A.shape[0])
                # boundary_points = abs(Y - h) < 1e-12        # Do this instead of " boundary_points = (Y == h) "
                # g[boundary_points] += ...
                # print("X", X)
                # print("Y", Y)

                cut = 1e-12

                boundary_points = abs(Y - h) < cut

                if TIME_DEBUG:
                    print(f"boundary points: {boundary_points}")
                # g[boundary_points] += (1 / h ** 2) * uexact(t,
                # X[boundary_points], Y[boundary_points] - h)
                g[boundary_points] += uexact(t, X[boundary_points],
                                             Y[boundary_points] - h)

                boundary_points = abs(Y - (1 - h)) < cut

                if TIME_DEBUG:
                    print(f"boundary points: {boundary_points}")
                # g[boundary_points] += (1 / h ** 2) * uexact(t,
                # X[boundary_points], Y[boundary_points] + h)
                g[boundary_points] += uexact(t, X[boundary_points],
                                             Y[boundary_points] + h)

                boundary_points = abs(X - h) < cut

                if TIME_DEBUG:
                    print(f"boundary points: {boundary_points}")
                # g[boundary_points] += (1 / h ** 2) * uexact(t,
                # X[boundary_points] - h, Y[boundary_points])
                g[boundary_points] += uexact(t, X[boundary_points] - h,
                                             Y[boundary_points])

                boundary_points = abs(X - (1 - h)) < cut

                if TIME_DEBUG:
                    print(f"boundary points: {boundary_points}")
                # g[boundary_points] += (1 / h ** 2) * uexact(t,
                # X[boundary_points] + h, Y[boundary_points])
                g[boundary_points] += uexact(t, X[boundary_points] + h,
                                             Y[boundary_points])

                # handle the corner cases SEPARATELY (will be counted )

                # looks ok

                # Backward Euler
                # Task: fill in the arguments to backward Euler
                # f(t,x,y)
                f_be = f(t, X, Y)

                # local_u = u_local[i - 1, rank]

                # Lecture 26, slides 22 and 23 -- need (1/h**2) factor for g?
                # ASSUMING time step is SMALL (for u_i-1 to be a good guess...)
                # euler_backward only returns the LOCAL portion (NOT including
                # halo regions)
                u_local[i][start_index:end_index] = \
                    euler_backward(A, u_local[i - 1][:], ht, f_be,
                                   (1 / h ** 2) * g, n_local_domain)

                # want f at
                # current time value

                # keep track of global only on process 0
                # no need to communicate since the values are calculated from the
                # function (designate process 0 to contain the exact values for
                # the entire grid...)
                # maybe this is where a good chunk of time is spent...with
                # the global version
                # wait, just don't keep track of uexact entirely...only
                # needed at the end to calculate the error...
                UE_GLOBAL_COMMUNICATE = True

                # also, to further reduce the time, the values of u are only
                # communicated ON THE LAST TIME STEP instead of EVERY TIME
                # STEP (one way communication since all the values are sent
                # to process 0) -- hopefully this works...

                # steps to reduce time
                # - parallelize the uexact (and only on last time step) --
                # ue_global
                # - send values of local u to process 0 only on last time
                # step -- u_global
                if rank == 0:
                    if PLOT_TIME_STEP:
                        u_global[i][start_index:end_index] = u_local[i][
                                                         start_index:end_index]
                    else:
                        if i == (nt - 1):
                            u_global[i][start_index:end_index] = \
                                u_local[i][start_index:end_index]

                    if i == (nt - 1):
                        if UE_GLOBAL_COMMUNICATE:
                            ue_global[i][start_index:end_index] = \
                                uexact(t, X, Y)[:n_local_domain]

                            # wait...there's the same instruction below...
                            # ue_global[i][start_index:end_index] = \
                            #     uexact(t, X, Y)[:n_local_domain]
                        else:
                            ue_global[i][:] = uexact(t, X_global, Y_global)
                else:
                    if UE_GLOBAL_COMMUNICATE:
                        if i == (nt - 1):
                            # if rank == (nprocs - 1):
                            #     # ue_local[i][start_index:end_index] = \
                            #     #     uexact(t, X, Y)[-n_local_domain:]
                            #     ue_local[i][:] = \
                            #         uexact(t, X, Y)[:]
                            # else:
                            #     # ue_local[i][start_index:end_index] = \
                            #     #     uexact(t, X, Y)[n_local_domain:-n_local_domain]
                            #     ue_local[i][:] = \
                            #         uexact(t, X, Y)[:]

                            ue_local[i][:] = \
                                uexact(t, X, Y)[:]

                if TIME_DEBUG:
                    print("g", g)
                    print(f"u (backward euler): {u_local[i, :]}")
                    print(f"u (exact): {ue_local[i, :]}")

                # added the communication under condition as well to reduce
                # total runtime
                # actually, we could just use the norm function above...with
                # allGather

                if not ERROR_NORM:
                    if nprocs > 1:
                        if rank != 0:
                            if rank == (nprocs - 1):
                                if PLOT_TIME_STEP:
                                    comm.Send(
                                        [u_local[i][-n_local_domain:], MPI.DOUBLE],
                                        dest=0,
                                        tag=77)
                                else:
                                    if i == (nt - 1):
                                        comm.Send([u_local[i][-n_local_domain:],
                                                   MPI.DOUBLE],
                                                  dest=0,
                                                  tag=77)

                                if UE_GLOBAL_COMMUNICATE:
                                    if i == (nt - 1):
                                        comm.Send(
                                            [ue_local[i][-n_local_domain:], MPI.DOUBLE],
                                            dest=0,
                                            tag=99)
                            else:
                                if PLOT_TIME_STEP:
                                    comm.Send([u_local[i][n_local_domain:
                                                          -n_local_domain],
                                               MPI.DOUBLE],
                                              dest=0,
                                              tag=77)
                                else:
                                    if i == (nt - 1):
                                        comm.Send([u_local[i][n_local_domain:
                                                              -n_local_domain],
                                                   MPI.DOUBLE],
                                                  dest=0,
                                                  tag=77)

                                if UE_GLOBAL_COMMUNICATE:
                                    if i == (nt - 1):
                                        comm.Send(
                                            [ue_local[i][n_local_domain:
                                                         -n_local_domain],
                                             MPI.DOUBLE],
                                            dest=0,
                                            tag=99)

                        if rank == 0:
                            # for now, just used a for loop to receive everything...
                            # try Gather instead of for loop with blocking
                            # receives...
                            #comm.Gather()

                            start_communication_time = time()

                            for rank_num in range(1, nprocs):
                                start_index_inner = rank_num * n_local_domain
                                end_index_inner = (rank_num + 1) * n_local_domain

                                if PLOT_TIME_STEP:
                                    comm.Recv([u_global[i]
                                               [start_index_inner:end_index_inner],
                                               MPI.DOUBLE],
                                              source=rank_num, tag=77)
                                else:
                                    if i == (nt - 1):
                                        comm.Recv([u_global[i]
                                                   [start_index_inner:
                                                    end_index_inner],
                                                   MPI.DOUBLE],
                                                  source=rank_num, tag=77)

                                if UE_GLOBAL_COMMUNICATE:
                                    if i == (nt - 1):
                                        comm.Recv([ue_global[i]
                                                   [start_index_inner:end_index_inner],
                                                   MPI.DOUBLE],
                                                  source=rank_num, tag=99)

                            end_communication_time = time()
                    else:
                        # remember to update u_global for 1 process
                        u_global[i][:] = u_local[i][:]

                if PLOT_TIME_STEP:
                    if rank == 0:
                        pyplot.figure(i)
                        pyplot.imshow(u_global[i, :].reshape(n, n), origin='lower',
                                      extent=(0, 1, 0, 1))
                        pyplot.colorbar()
                        pyplot.xlabel('X')
                        pyplot.ylabel('Y')
                        pyplot.title(f"Solution, i={i}")
                        pyplot.savefig(
                            f"{parallel_plots_dir_current}solution_{i}.png")
                        pyplot.close(pyplot.figure(i))

            # end time
            if rank == 0:
                end_time = time()

                total_time = end_time - start_time
                timings_array[timing_index] = total_time

                if not ERROR_NORM:
                    communication_time = end_communication_time - \
                                         start_communication_time

                sys.stderr.write(f"timing {timing_index}: {total_time}\n")

                if not ERROR_NORM:
                    sys.stderr.write(f"communication time -- last time step to "
                                     f"calculate error: {communication_time}\n")
                    sys.stderr.write(f"percentage of communication: "
                                     f"{(communication_time / total_time) * 100}\n")

                # stores the timings in a file with format
                # {num_processes} {timing}
                with open(f"{parallel_root}timings.txt", "a+") as time_file:
                    time_file.write(f"{nprocs} {total_time}\n")

                    # make sure to close file after writing...
                    # for some reason, the file won't be transferred over if
                    # not closed when the walltime has been reached and the
                    # job is killed...
                    time_file.close()

            if ERROR_NORM:
                if rank == 0:
                    error_local = (u_local[-1][:n_local_domain] -
                                   ue_local[-1][:n_local_domain]).reshape(-1, )
                elif rank == (nprocs - 1):
                    error_local = (u_local[-1][-n_local_domain:] -
                                   ue_local[-1][-n_local_domain:]).reshape(-1, )
                else:
                    error_local = (u_local[-1][n_local_domain:
                                               -n_local_domain] -
                                   ue_local[-1][n_local_domain:
                                                -n_local_domain]).reshape(-1, )

                # sys.stderr.write(f"rank: {rank}\n")
                # sys.stderr.write(f"error local: {error_local}\n")

                enorm = norm(error_local, comm)

            if rank == 0:
                if FINAL_DEBUG:
                    print(f"final u (backward euler): {u_global[-1, :]}")
                    print(f"final u (exact) array: {ue_global[-1, :]}")

                    # changed from X, X to X_global, Y_global (above)
                    print(
                        f"final u (exact) function: {uexact(T, X_global, Y_global)}")

                    print(f"ue_global: {ue_global}")

                if ERROR_NORM:
                    #enorm = linalg.norm(e) * h
                    enorm = enorm * h

                    sys.stderr.write(
                        "Nt, N, Error is:  " + str(nt) + ",  " + str(n)
                        + ",  " + str(enorm) + "\n")
                    error.append(enorm)
                else:
                    # Compute L2-norm of the error at final time
                    e = (u_global[-1, :] - ue_global[-1, :]).reshape(-1, )
                    enorm = linalg.norm(
                        e) * h  # Task: compute the L2 norm over space-time
                    # here.  In serial this is just one line.  In parallel...
                    # Parallel task: In parallel, write a helper function to compute the norm of "e" in parallel

                    # change print to sys.stderr.write
                    sys.stderr.write("Nt, N, Error is:  " + str(nt) + ",  " + str(n)
                                     + ",  " + str(enorm) + "\n")
                    error.append(enorm)

            # You can turn this on to visualize the solution.  Possibly helpful for debugging.
            # Only works in serial.  Parallel visualizations will require that you stitch together
            # the solution from each processor before one single processor generates the graphic.
            # But, you can use the imshow command for your report graphics, as done below.
            ORIGINAL = False
            if rank == 0:
                if PLOT_TIME_STEP:  # to reduce time even further, ignore
                    # creating the plots and communication of local portion
                    # of u above...
                    pyplot.figure(-1)
                    pyplot.imshow(u_global[0, :].reshape(n, n), origin='lower',
                                  extent=(0, 1, 0, 1))
                    pyplot.colorbar()
                    pyplot.xlabel('X')
                    pyplot.ylabel('Y')
                    pyplot.title("Initial Condition")
                    pyplot.savefig(f"{parallel_root_current}solution_initial.png")
                    pyplot.close(pyplot.figure(-1))

                    # pyplot.figure(10)
                    # pyplot.imshow(u[5,:].reshape(n,n))
                    # pyplot.colorbar()
                    # pyplot.xlabel('X')
                    # pyplot.ylabel('Y')
                    # pyplot.title("Solution at final time")

                    pyplot.figure(-12)
                    # print(f"Nt shape: {Nt_values.shape}")
                    # print(f"mid time index: {Nt_values[0] // 2}")

                    if ORIGINAL:
                        pyplot.imshow(u_global[Nt_values[0] // 2, :].reshape(n, n))
                    else:
                        pyplot.imshow(u_global[Nt_values[0] // 2, :].reshape(n, n),
                                      origin='lower', extent=(0, 1, 0, 1))
                    pyplot.colorbar()
                    pyplot.xlabel('X')
                    pyplot.ylabel('Y')
                    pyplot.title("Solution at mid time")
                    pyplot.savefig(f"{parallel_root_current}solution_mid.png")
                    pyplot.close(pyplot.figure(-12))

                    pyplot.figure(-99)
                    if ORIGINAL:
                        pyplot.imshow(
                            uexact(t0 + (Nt_values[0] // 2) * ht, X_global,
                                   Y_global).reshape(n, n))
                    else:
                        pyplot.imshow(uexact(t0 + (Nt_values[0] // 2) * ht,
                                             X_global, Y_global)
                                      .reshape(n, n), origin='lower', extent=(0, 1,
                                                                              0, 1))
                    pyplot.colorbar()
                    pyplot.xlabel('X')
                    pyplot.ylabel('Y')
                    pyplot.title("Exact Solution at mid time")
                    pyplot.savefig(f"{parallel_root_current}exact_mid.png")
                    pyplot.close(pyplot.figure(-99))

                    # import pdb; pdb.set_trace()

                    # pyplot.figure(11)
                    # pyplot.imshow(u[-22,:].reshape(n,n))
                    # pyplot.colorbar()
                    # pyplot.xlabel('X')
                    # pyplot.ylabel('Y')
                    # pyplot.title("Solution at final time")

                    pyplot.figure(-3)
                    if ORIGINAL:
                        pyplot.imshow(u_global[-1, :].reshape(n, n))
                    else:
                        pyplot.imshow(u_global[-1, :].reshape(n, n),
                                      origin='lower', extent=(0, 1, 0, 1))
                    pyplot.colorbar()
                    pyplot.xlabel('X')
                    pyplot.ylabel('Y')
                    pyplot.title("Solution at final time")
                    pyplot.savefig(f"{parallel_root_current}solution_final.png")
                    pyplot.close(pyplot.figure(-3))

                    pyplot.figure(-4)
                    if ORIGINAL:
                        pyplot.imshow(uexact(T, X_global,
                                             Y_global).reshape(n, n))
                    else:
                        pyplot.imshow(uexact(T, X_global,
                                             Y_global).reshape(n, n),
                                      origin='lower', extent=(0, 1, 0, 1))
                    pyplot.colorbar()
                    pyplot.xlabel('X')
                    pyplot.ylabel('Y')
                    pyplot.title("Exact Solution at final time")
                    pyplot.savefig(f"{parallel_root_current}exact_final.png")
                    pyplot.close(pyplot.figure(-4))

                    # pyplot.show()

    if rank == 0:
        with open(f"{parallel_root}timings.txt", "a+") as time_file:
            time_file.write(f"min time: {min(timings_array)}\n")
            time_file.close()

    # Plot convergence
    # Need to plot the weak scaling MANUALLY after all runs are completed in
    # a separate Python script
    # if rank == 0:
    #     if True:
    #         # When generating this plot in parallel, you should have only rank=0
    #         # save the graphic to a .PNG
    #         pyplot.figure(-999)
    #         pyplot.loglog(1. / N_values, 1. / N_values ** 2, '-ok')
    #         pyplot.loglog(1. / N_values, array(error), '-sr')
    #         pyplot.tick_params(labelsize='large')
    #         pyplot.xlabel(r'Spatial $h$', fontsize='large')
    #         pyplot.ylabel(r'$||e||_{L_2}$', fontsize='large')
    #         pyplot.legend(['Ref Quadratic', 'Computed Error'], fontsize='large')
    #
    #         # TODO: save error plot -- change to tight for final...
    #         pyplot.savefig(f'{parallel_root_current}error.png', dpi=500,
    #                        format='png',
    #                        pad_inches=0.0, )
    #         # pyplot.savefig(f'{parallel_root}error.png', dpi=500, format='png',
    #         #                bbox_inches='tight', pad_inches=0.0,)
    #
    #         # TODO: comment out the show plots for final...
    #         # pyplot.show()
    #
    #         pyplot.close(pyplot.figure(-999))

# TODO: 1 process, no communication...
# TODO: another directory for plots (separate serial and parallel...)
# TODO: error with more than 1 process -- multiple processes (n_local_domain...)

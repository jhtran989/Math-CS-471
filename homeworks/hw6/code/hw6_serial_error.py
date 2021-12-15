import numpy
from numpy import *
from matplotlib import pyplot 
from poisson import poisson
# speye generates a sparse identity matrix
from scipy.sparse import eye as speye
from scipy.sparse.linalg import splu

import os


global_maxiter = 400  # go through code and refactor
global_tol = 1e-10  # 1e-10, 1e-15

print(f"tol: {global_tol}")

# Plots Stuff
serial_root = f"serial/"
serial_plots_dir = f"{serial_root}plots/"

os.makedirs(serial_plots_dir, exist_ok=True)

# DEBUG Stuff
PLOT_TIME_STEP = False
FINAL_DEBUG = False

# print convergence check
CONVERGENCE_CHECK = True

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
def uexact(t,x,y):
    # Task: fill in exact solution
    #return sin(pi*t)*sin(pi*x)*sin(pi*y) # this is the exact solution
    return cos(pi*t)*cos(pi*x)*cos(pi*y)
    #return (t-0.9)*(x**2)*(y**2)

def f(t,x,y):
    # Forcing term
    # This should equal u_t - u_xx - u_yy
    
    # Task: fill in forcing term
    #return pi*(cos(pi*t))*sin(pi*x)*sin(pi*y) + 2*pi*pi*uexact(t, x, y)
    # this is f, change for new cos function
    return pi*(-sin(pi*t))*cos(pi*x)*cos(pi*y) + 2*pi*pi*uexact(t, x, y)
    #return 1 - (((t - 0.9) * 2 * (y ** 2)) + ((t - 0.9) * 2 * (x ** 2)))
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
def jacobi(A, b, x0, tol, maxiter):
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
    '''

    # This useful function returns an array containing diag(A)
    D = A.diagonal()
    #print(f"Diagonal: {D}")
    D_inv = 1/D    #linalg.inv(D)
    #print(f"Diagonal inverse: {D_inv}")

    # compute initial residual norm
    r0 = ravel(b - A*x0)
    r0 = sqrt(dot(r0, r0))  # this is the init residual
    #print(f"r0: {r0}")

    I = speye(A.shape[0], format='csr')
    # Start Jacobi iterations 
    # Task in serial: implement Jacobi method and halting tolerance based on the residual norm
    
    # Task in parallel: extend the matrix-vector multiply to the parallel setting.  
    #                   Additionally, you'll need to compute a norm of the residual in parallel.
    x = zeros((maxiter+1, A.shape[0]))  # only need first and last
    x[0] = x0
    last_i = 0
    for i in range(maxiter):
        # << Jacobi algorithm goes here >>
        # Lecture 26, Slide 23

        # old
        x[i+1] = (x[i] - (A*x[i])*D_inv) + D_inv*b

        # new -- A is G in slides
        # use numpy.dot() instead of *
        # print(f"dot D_inv and A: {numpy.dot(D_inv, A)}")
        #
        # #inner_part = I - numpy.dot(D_inv, A)
        # inner_part = I - D_inv*A
        # first_term = numpy.dot(inner_part, x[i])
        # second_term = numpy.dot(D_inv, b)
        #
        # print(f"inner_part shape: {inner_part.shape}")
        # print(f"x[i] shape: {x[i].shape}")
        # print(f"first term: {first_term.shape}")
        # print(f"second term: {second_term.shape}")
        #
        # x[i+1] = first_term + second_term

        rk = b - A*x[i+1]
        # print(f"before ravel:")
        #print(f"rk: {rk}")
        # print(f"rk shape: {rk.shape}")
        rk = ravel(rk)  # does nothing...
        # print(f"after ravel:")
        # print(f"rk: {rk}")
        # print(f"rk shape: {rk.shape}")
        rk = sqrt(dot(rk, rk))
        #print(f"rk: {rk}")
        # (I - D_inv*A)*x_k+D_in*b
        # x_k-1 + x_k+1/2
        last_i = i

        # print(f"r0: {r0}")
        # print(f"rk: {rk}")
        if rk/r0 <= tol:
            if CONVERGENCE_CHECK:
                print("did converge, i = ", i)

            break

    # Task: Print if Jacobi did not converge. In parallel, only rank 0 should print.
    if rk/r0 > tol:
        if CONVERGENCE_CHECK:
            print("did not converge")

    #print("x", x.shape)
    return x[last_i+1:last_i+2] # this is good


def euler_backward(A, u, ht, f, g):
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

    '''
    #print("g", g)
    # Task: Form the system matrix for backward Euler
    I = speye(A.shape[0], format='csr')
    G = I - ht*A
    b = u + ht*g + ht*f # G*u # PP 26 pg 22   fix this
    #Ainv.solve(eye(A.shape[0]) - ht*A)*(u+ht*f+ht*f)
    # Task: return solution from Jacobi, which takes a time-step forward in time by "ht"
    # jacobi(A, b, x0, tol, maxiter):
    return jacobi(G, b, u, global_tol, global_maxiter)
    #Ainv = splu(G)
    #return Ainv.solve(b)
    #return G_inv.solve() # exact solve from lab to check jacobi, if not fixed, than issue is not in jacobi


# Helper function provided by instructor for debugging.  See how matvec_check
# is used below.
def matvec_check(A, X, Y, N, comm):
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
    '''

    nprocs = comm.size
    my_rank = comm.Get_rank()
    
    o = ones((A.shape[0],))
    oo = matrix_vector(A, o, N, comm)
    if my_rank != 0:
        oo = oo[N:]
        X = X[N:]
        Y = Y[N:]
    if my_rank != (nprocs-1):
        oo = oo[:-N]
        X = X[:-N]
        Y = Y[:-N]
    import sys 
    for i in range(oo.shape[0]):
        sys.stderr.write("X,Y: (%1.2e, %1.2e),  Ouput: %1.2e\n"%(X[i], Y[i], oo[i]))


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
    #Nt_values = array([8, 8*4, 8*4*4, 8*4*4*4])
    #N_values = array([8, 16, 32, 64 ])
    #T = 0.5

    # Changed for our special case (second function above)
    Nt_values = array([12 * (4 ** i) for i in range(4)])  # 8*4 -> 100
    N_values = array([8 * (2 ** i) for i in range(4)])  # 16
    #
    # print(f"N time values: {Nt_values}")
    # print(f"N values: {N_values}")

    # One very small problem for debugging
    # Nt_values = array([8])  # 8*4 -> 100
    # N_values = array([8])  # 16
    # T = 0.5  # 0.5

    # Changed for our special case (second function above)
    # Nt_values = array([12 * (4 ** 2)]) # 8*4 -> 100
    # N_values = array([8 * (2 ** 2)])  # 16
    T = 0.75  # 0.5

    print(f"number of processes: {nprocs}\n")
    print(f"Nt values: {Nt_values}\n")
    print(f"N values: {N_values}\n")
    print(f"T: {T}\n")

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

    # Define list to contain the discretization error from each problem size
    error = []

    # Begin loop over various numbers of time points (nt) and spatial grid sizes (n)
    for (nt, n) in zip(Nt_values, N_values):

        # Declare time step size
        t0 = 0.0
        ht = (T - t0)/float(nt-1)

        # Declare spatial grid size.  Note that we divide by (n + 1) because we are
        # accounting for the boundary points, i.e., we really have n+2 total points
        h = 1.0 / (n+1.0)

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
        pts = linspace(h, 1-h, n)
        X,Y = meshgrid(pts, pts)
        X = X.reshape(-1,)
        Y = Y.reshape(-1,)
        print("pts", pts)

        # Declare spatial discretization matrix
        # Task: what dimension should A be?  remember the spatial grid is from
        #       [h, 2h, ..., 1-h] x [h, 2h, ..., 1-h]
        #       Pass in the right size to poisson.
        # Task in parallel: Adjust the size of A, that is A will be just a processor's
        #                   local part of A, similar to HW4
        # in HW4 poisson((end_halo - start_halo, n), format='csr')

        sizex = n
        sizey = n

        A = poisson((sizey, sizex), format='csr')

        # Task: scale A by the grid size
        A = (1/h**2)*(A)


        # Declare initial condition
        #   This initial condition obeys the boundary condition.
        #print("X.shape", X.shape)
        #print("Y.shape", Y.shape)
        u0 = uexact(0, X, Y)


        # Declare storage
        # Task: Declare "u" and "ue".  What sizes should they be?  "u" will store the
        #       numerical solution, and "ue" will store the exact solution.
        # Task in parallel: Adjust the sizes to be only for this processor's
        #                   local portion of the domain.
        #print("maxiter", maxiter)
        #print("A.size[0]", A.shape[0])

        A_shape = int((A.shape[0]))
        print(A_shape)
        u = zeros((nt, A_shape))
        ue = zeros((nt, A_shape))

        # Set initial condition
        print("u", u.shape)
        print("u0", u0)


        u[0,:] = u0 #u0[-1:]  # starts at exact solution
        ue[0,:] = u0 #[-1:]

        pyplot.figure(0)
        pyplot.imshow(u[0, :].reshape(n, n), origin='lower',
                      extent=(0, 1, 0, 1))
        pyplot.colorbar()
        pyplot.xlabel('X')
        pyplot.ylabel('Y')
        pyplot.title(f"Solution, i={0}")
        pyplot.savefig(f"{serial_plots_dir}solution_{0}.png")

        # Testing harness for parallel part: Only comment-in and run for the smallest
        # problem size of 8 time points and an 8x8 grid
        #   - Assumes specific structure to your mat-vec multiply routine
        #    (described above in comments)
        #matvec_check( (h**2)*A, X, Y, n-2, comm)

        TIME_DEBUG = False
        #t = t0

        # Run time-stepping over "nt" time points
        for i in range(1,nt):
            t = t0 + i*ht
            # ERROR - DON'T use t0 (only initial iteration)

            if TIME_DEBUG:
                print(f"------------------------------")
                print("t", t)
                print("i (time)", i)

            # Task: We need to store the exact solution so that we can compute the error
            #print("ue.shape", ue.shape)
            #print("uexact.shape", uexact(t, X, Y).shape)
            ue[i,:] = uexact(t, X, Y)

            # Task: Compute boundary contributions for the current time value of i*ht
            #       Different from HW4, need to account for numeric error, hence "1e-12" and not "0"
            g = zeros((A.shape[0],))
            #print("a_shape[0]", A.shape[0])
            #boundary_points = abs(Y - h) < 1e-12        # Do this instead of " boundary_points = (Y == h) "
            #g[boundary_points] += ...
            #print("X", X)
            #print("Y", Y)

            cut = 1e-12

            boundary_points = abs(Y - h) < cut

            if TIME_DEBUG:
                print(f"boundary points: {boundary_points}")
            #g[boundary_points] += (1 / h ** 2) * uexact(t,
            # X[boundary_points], Y[boundary_points] - h)
            g[boundary_points] += uexact(t, X[boundary_points],
            Y[boundary_points] - h)

            boundary_points = abs(Y-(1-h)) < cut

            if TIME_DEBUG:
                print(f"boundary points: {boundary_points}")
            #g[boundary_points] += (1 / h ** 2) * uexact(t,
            # X[boundary_points], Y[boundary_points] + h)
            g[boundary_points] += uexact(t, X[boundary_points],
            Y[boundary_points] + h)

            boundary_points = abs(X - h) < cut

            if TIME_DEBUG:
                print(f"boundary points: {boundary_points}")
            #g[boundary_points] += (1 / h ** 2) * uexact(t,
            # X[boundary_points] - h, Y[boundary_points])
            g[boundary_points] += uexact(t, X[boundary_points] - h,
            Y[boundary_points])

            boundary_points = abs(X - (1-h)) < cut

            if TIME_DEBUG:
                print(f"boundary points: {boundary_points}")
            #g[boundary_points] += (1 / h ** 2) * uexact(t,
            # X[boundary_points] + h, Y[boundary_points])
            g[boundary_points] += uexact(t, X[boundary_points] + h,
            Y[boundary_points])

            # handle the corner cases SEPARATELY (will be counted )

            # looks ok

        # Backward Euler
            # Task: fill in the arguments to backward Euler
            # f(t,x,y)
            f_be = f(t, X, Y)

            # Lecture 26, slides 22 and 23 -- need (1/h**2) factor for g?
            # ASSUMING time step is SMALL (for u_i-1 to be a good guess...)
            u[i,:] = euler_backward(A, u[i-1,:], ht, f_be, (1/h**2) * g)
            # want f at
            # current time value

            if TIME_DEBUG:
                print("g", g)
                print(f"u (backward euler): {u[i, :]}")
                print(f"u (exact): {ue[i, :]}")

            if PLOT_TIME_STEP:
                pyplot.figure(i)
                pyplot.imshow(u[i, :].reshape(n, n), origin='lower',
                              extent=(0, 1, 0, 1))
                pyplot.colorbar()
                pyplot.xlabel('X')
                pyplot.ylabel('Y')
                pyplot.title(f"Solution, i={i}")
                pyplot.savefig(f"{serial_plots_dir}solution_{i}.png")
                pyplot.close(pyplot.figure(i))

        if FINAL_DEBUG:
            # Compute L2-norm of the error at final time
            print(f"final u (backward euler): {u[-1,:]}")
            print(f"final u (exact): {ue[-1, :]}")

        e = (u[-1,:] - ue[-1,:]).reshape(-1,)
        enorm = linalg.norm(e) * h # Task: compute the L2 norm over space-time
        # here.  In serial this is just one line.  In parallel...
        # Parallel task: In parallel, write a helper function to compute the norm of "e" in parallel

        print("Nt, N, Error is:  " + str(nt) + ",  " + str(n) + ",  " + str(enorm))
        error.append(enorm)


        # You can turn this on to visualize the solution.  Possibly helpful for debugging.
        # Only works in serial.  Parallel visualizations will require that you stitch together
        # the solution from each processor before one single processor generates the graphic.
        # But, you can use the imshow command for your report graphics, as done below.
        ORIGINAL = False
        if True:
            pyplot.figure(-1)
            pyplot.imshow(u[0,:].reshape(n,n), origin='lower', extent=(0, 1,
                                                                        0, 1))
            pyplot.colorbar()
            pyplot.xlabel('X')
            pyplot.ylabel('Y')
            pyplot.title("Initial Condition")
            pyplot.savefig(f"{serial_root}solution_initial.png")


            #pyplot.figure(10)
            #pyplot.imshow(u[5,:].reshape(n,n))
            #pyplot.colorbar()
            #pyplot.xlabel('X')
            #pyplot.ylabel('Y')
            #pyplot.title("Solution at final time")

            pyplot.figure(-12)
            #print(f"Nt shape: {Nt_values.shape}")
            print(f"mid time index: {Nt_values[0] // 2}")

            if ORIGINAL:
                pyplot.imshow(u[Nt_values[0] // 2, :].reshape(n, n))
            else:
                pyplot.imshow(u[Nt_values[0] // 2,:].reshape(n,n), origin='lower', extent=(0, 1,
                                                                        0, 1))
            pyplot.colorbar()
            pyplot.xlabel('X')
            pyplot.ylabel('Y')
            pyplot.title("Solution at mid time")
            pyplot.savefig(f"{serial_root}solution_mid.png")

            pyplot.figure(-99)
            if ORIGINAL:
                pyplot.imshow(
                    uexact(t0 + (Nt_values[0] // 2) * ht, X, Y).reshape(n,
                                                                        n))
            else:
                pyplot.imshow(uexact(t0 + (Nt_values[0] // 2) * ht, X, Y).reshape(n,
                                                                           n), origin='lower', extent=(0, 1,
                                                                        0, 1))
            pyplot.colorbar()
            pyplot.xlabel('X')
            pyplot.ylabel('Y')
            pyplot.title("Exact Solution at mid time")
            pyplot.savefig(f"{serial_root}exact_mid.png")

            #import pdb; pdb.set_trace()

            #pyplot.figure(11)
            #pyplot.imshow(u[-22,:].reshape(n,n))
            #pyplot.colorbar()
            #pyplot.xlabel('X')
            #pyplot.ylabel('Y')
            #pyplot.title("Solution at final time")


            pyplot.figure(-3)
            if ORIGINAL:
                pyplot.imshow(u[-1, :].reshape(n, n))
            else:
                pyplot.imshow(u[-1,:].reshape(n,n), origin='lower', extent=(0, 1,
                                                                        0, 1))
            pyplot.colorbar()
            pyplot.xlabel('X')
            pyplot.ylabel('Y')
            pyplot.title("Solution at final time")
            pyplot.savefig(f"{serial_root}solution_final.png")

            pyplot.figure(-4)
            if ORIGINAL:
                pyplot.imshow(uexact(T, X, Y).reshape(n, n))
            else:
                pyplot.imshow(uexact(T,X,Y).reshape(n,n), origin='lower', extent=(0, 1,
                                                                        0, 1))
            pyplot.colorbar()
            pyplot.xlabel('X')
            pyplot.ylabel('Y')
            pyplot.title("Exact Solution at final time")
            pyplot.savefig(f"{serial_root}exact_final.png")

            #pyplot.show()


    # Plot convergence
    if True:
        # When generating this plot in parallel, you should have only rank=0
        # save the graphic to a .PNG
        pyplot.figure(-999)
        pyplot.loglog(1./N_values, 1./N_values**2, '-ok')
        pyplot.loglog(1./N_values, array(error), '-sr')
        pyplot.tick_params(labelsize='large')
        pyplot.xlabel(r'Spatial $h$', fontsize='large')
        pyplot.ylabel(r'$||e||_{L_2}$', fontsize='large')
        pyplot.legend(['Ref Quadratic', 'Computed Error'], fontsize='large')

        # TODO: save error plot -- change to tight for final...
        pyplot.savefig(f'{serial_root}error.png', dpi=500, format='png',
                       pad_inches=0.0, )
        # pyplot.savefig(f'{serial_root}error.png', dpi=500, format='png',
        #                bbox_inches='tight', pad_inches=0.0,)

        # TODO: comment out the show plots for final...
        #pyplot.show()
        #pyplot.savefig('error.png', dpi=500, format='png', bbox_inches='tight', pad_inches=0.0,)





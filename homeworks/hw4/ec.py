from numpy import *
from scipy import sparse
import time
from threading import Thread
from matplotlib import pyplot
from poisson import poisson
from check_matvec import check_matvec
import sys
from timeit import default_timer as timer


DEBUG = False


def L2norm(e, h):
    '''
    Take L2-norm of e
    '''
    # ensure e has a compatible shape for taking a dot-product
    e = e.reshape(-1, )

    # Task:
    # Return the L2-norm, i.e., the square roof of the integral of e^2
    # Assume a uniform grid in x and y, and apply the midpoint rule.
    # Assume that each grid point represents the midpoint of an equally sized region
    return linalg.norm(e) * (h)  ### good


def get_2d_case_output(offset, output, num_elements_row, POSITION):
    # output = output

    if POSITION == "t":
        output = output[num_elements_row:]
    elif POSITION == "m":
        output = output[num_elements_row:-num_elements_row]
    elif POSITION == "b":
        output = output[:-num_elements_row]
    else:
        sys.stderr.write(f"INVALID POSITION: please choose from 't', 'm', "
                         f"or 'b'")
        exit(99)

    # if offset[0] != 0 and offset[1] == 0:
    #     output = output[num_elements_row:]
    # elif offset[0] == 0 and offset[1] != 0:
    #     output = output[:-num_elements_row]
    # elif offset[0] != 0 and offset[1] != 0:
    #     output = output[num_elements_row:-num_elements_row]
    # else:
    #     # if offset[0] == 0 and offset[1] == 0:
    #     # do nothing
    #     output = output

    num_rows = len(output) // num_elements_row

    if DEBUG:
        sys.stderr.write(f"num rows: {num_rows}\n")
        sys.stderr.write(f"offset: {offset}\n")
        sys.stderr.write(f"position: {POSITION}\n")

    temporary_output = []
    for i in range(0, num_rows):
        start_output = (i * num_elements_row) + offset[0]
        end_output = ((i + 1) * num_elements_row) + offset[1]

        temporary_output.append(output[start_output:end_output])
        # temporary_output.append(output[:-1])

    output = list(concatenate(temporary_output).flat)

    if DEBUG:
        sys.stderr.write(f"(in function) output length: {len(output)}\n")
        sys.stderr.write(f"(in function) output: {output}\n")

    return output  # doesn't pass by reference?


def compute_fd_2d(n, nt, k, f, fpp_num): # new
    '''
    Compute the numeric second derivative of the function 'f' with a
    threaded matrix-vector multiply.

    Input
    -----
    n   <int>       :   Number of grid points in x and y for global problem
    nt  <int>       :   Number of threads
    k   <int>       :   My thread number
    f   <func>      :   Function to take second derivative of
    fpp_num <array> :   Global array of size n**2


    Output
    ------
    fpp_num will have this thread's local portion of the second derivative
    written into it


    Notes
    -----
    We do a 1D domain decomposition.  Each thread 'owns' the k*(n/nt) : (k+1)*(n/nt) rows
    of the domain.

    For example,
    Let the global points in the x-dimension be [0, 0.33, 0.66, 1.0]
    Let the global points in the y-dimension be [0, 0.33, 0.66, 1.0]
    Let the number of threads be two (nt=2)

    Then for the k=0 case (for the 0th thread), the domain rows  'owned' are
    y = 0,    and x = [0, 0.33, 0.66, 1.0]
    y = 0.33, and x = [0, 0.33, 0.66, 1.0]

    Then for the k = 1, case, the domain rows 'owned' are
    y = 0.66, and x = [0, 0.33, 0.66, 1.0]
    y = 1.0,  and x = [0, 0.33, 0.66, 1.0]

    We assume that n/nt divides evenly.

    '''

    # Task:
    # Compute start, end
    #
    # These indices allow you to index into arrays and grab only this thread's
    # portion.  For example, using the y = [0, 0.33, 0.66, 1.0] example above,
    # and considering thread 0, will yield start = 0 and end = 2, so that
    # y[start:end] --> [0, 0.33]
    #start = int(k * (n / nt))

    sys.stderr.write(f"--------------------------------------\n")
    
    dimension_length = int(sqrt(nt))

    start_x = int(int(k % dimension_length) * n / dimension_length)
    start_y = int(int(k // dimension_length) * n / dimension_length)
    # start_x = int(int(k % dimension_length))
    # start_y = int(int(k // dimension_length))

    #  <first domain row owned by thread k,
    # cast as integer>
    #end = int((k + 1) * (n / nt))  #  <first domain row owned by thread k+1, cast as
    end_y = int((int(k // dimension_length) + 1) * n / dimension_length)
    end_x = int((int(k % dimension_length) + 1) * n / dimension_length)
    # end_y = int((int(k // dimension_length) + 1))
    # end_x = int((int(k % dimension_length) + 1))

    x_ind = k % dimension_length
    y_ind = k // dimension_length

    if DEBUG:
        sys.stderr.write(f"number of threads: {nt}\n")
        sys.stderr.write(f"dimension length: {dimension_length}\n")
        sys.stderr.write(f"n: {n}\n")
        sys.stderr.write(f"start_x: {start_x}\n")
        sys.stderr.write(f"end_x: {end_x}\n")
        sys.stderr.write(f"start_y: {start_y}\n")
        sys.stderr.write(f"end_y: {end_y}\n")
        sys.stderr.write(f"x index: {x_ind}\n")
        sys.stderr.write(f"y index: {y_ind}\n")

    # integer>
    # good
    #sys.stderr.write(f"start: {start}")
    #sys.stderr.write(f"end: {end}")

    # Task:
    # Compute start_halo, and end_halo
    #
    # These values are the same as start and end, only they are expanded to
    # include the halo region.
    #
    # Halo regions essentially expand a thread's local domain to include enough
    # information from neighboring threads to carry out the needed computation.
    # For the above example, that means
    #   - Including the y=0.66 row of points for the k=0 case
    #     so that y[start_halo : end_halo] --> [0, 0.33, 0.66]
    #   - Including the y=0.33 row of points for the k=1 case
    #     so that y[start_halo : end_halo] --> [0.33, 0.66, 1.0]
    #   - Note that for larger numbers of threads, some threads
    #     will have halo regions including domain rows above and below.
    #if k != 0:
    #    start_halo = start - 1
    #else:
    #    start_halo = start
    # start_halo = <start - 1, unless k == 0>
    #if k != (nt - 1):
    #    end_halo = end + 1
    #else:
    #    end_halo = end

    # y direction
    if y_ind > 0:
        start_halo_y = start_y - 1
    else:
        start_halo_y = start_y
    if y_ind < (dimension_length - 1):
        end_halo_y = end_y + 1
    else:
        end_halo_y = end_y

    # x direction
    if x_ind > 0:
        start_halo_x = start_x - 1
    else:
        start_halo_x = start_x
    if x_ind < (dimension_length - 1):
        end_halo_x = end_x + 1
    else:
        end_halo_x = end_x

    if DEBUG:
        sys.stderr.write(f"start_halo_x: {start_halo_x}\n")
        sys.stderr.write(f"end_halo_x: {end_halo_x}\n")
        sys.stderr.write(f"start_halo_y: {start_halo_y}\n")
        sys.stderr.write(f"end_halo_y: {end_halo_y}\n")


    # good
    # end_halo = <end + 1, unless k == (nt-1)>

    # Construct local CSR matrix.  Here, you're given that function in poisson.py
    # This matrix will contain the extra halo domain rows

    A = poisson((end_halo_y - start_halo_y, end_halo_x - start_halo_x), format='csr')
    h = 1. / (n - 1)
    A *= 1 / h ** 2

    # Task:
    # Inspect a row or two of A, and verify that it's the correct 5 point stencil
    # You can print a few rows of A, with print(A[k,:])
    # < add statement to inspect a row or two of A >

    #sys.stderr.write(A[k, :])  # check

    # Task:
    # Construct a grid of evenly spaced points over this thread's halo region
    #
    # x_pts contains all of the points in the x-direction in this thread's halo region
    #x_pts = linspace(0, 1, n)
    x_pts = linspace(start_halo_x * h, (end_halo_x-1) * h, (end_halo_x-start_halo_x))
    #
    # y_pts contains all of the points in the y-direction for this thread's halo region
    # For the above example and thread 1 (k=1), this is y_pts = [0.33, 0.66, 1.0]
    y_pts = linspace(start_halo_y * h, (end_halo_y-1) * h, (end_halo_y-start_halo_y))  # good
    #sys.stderr.write(f"x_pts: {x_pts.shape}\n")
    #sys.stderr.write(f"y_pts: {y_pts.shape}\n")
    # Task:
    # There is no coding to do here, but examime how meshgrid works and
    # understand how it gives you the correct uniform grid.
    X, Y = meshgrid(x_pts, y_pts)
    X = X.reshape(-1, )
    Y = Y.reshape(-1, )

    # I have had many headaches plotting higher dimensional mesh grids

    # Task:
    # Compute local portion of f by using X and Y
    # x_region = X[start_halo:end_halo]
    # y_region = Y[start_halo:end_halo]
    # print("x_region = ", x_region.size)
    # print("y_region =", y_region.size)
    if DEBUG:
        sys.stderr.write(f"X shape: {X.shape}\n")
        sys.stderr.write(f"X: {X}\n")
        sys.stderr.write(f"Y shape: {Y.shape}\n")
        sys.stderr.write(f"Y: {Y}\n")

    f_vals = f(X, Y)  # < f evaluated at X and Y > # good

    if DEBUG:
        sys.stderr.write(f"A section: \n")
        sys.stderr.write(f"{A.todense()}\n")
        sys.stderr.write(f"fvals: \n")
        sys.stderr.write(f"{f_vals}\n")

    # Task:
    # Compute the correct range of output values for this thread
    # print("A = ", A.size)
    # print("f_vals", f_vals.size)
    # A= A.todense()
    #sys.stderr.write(f"A: {A.todense()}\n")
    #sys.stderr.write(f"f_vals: {f_vals.shape}\n")

    output = A * f_vals

    if DEBUG:
        sys.stderr.write(f"initial output: {output}\n")
    #print("output_init", output)
    # print(f"output: {output}")
    # print(f"output shape: {output.shape}")
    #start_output = int(k * (n**2 / nt))
    #end_output = int((k + 1) * (n**2 / nt))
    #sys.stderr.write(f"start_output: {start_output}\n")
    #sys.stderr.write(f"end_output: {end_output}\n")
    #sys.stderr.write(f"k: {k}\n")
    #if k != 0 and k != (nt-1):
    #    output = output[n:-n]  # good
    #if k == 0 and nt == 1:
    #    output = output    # good
    #if k == 0 and nt > 1:
    #    output = output[:-n]    # good
    #if k == (nt-1) and k != 0:
    #    output = output[n:]   # good

    # ***
    # ***
    # ***

    num_elements_row = end_halo_x - start_halo_x
    num_elements_column = end_halo_y - start_halo_y

    if DEBUG:
        sys.stderr.write(f"initial num elements row: {num_elements_row}\n")

    # need to check for nt == 1
    if nt == 1:
        if k == 0:
            sys.stderr.write(f"Running 1 Thread...\n")

            # not needed -- finished
            #output = output
        else:
            sys.stderr.write(f"ERROR: WRONG K VALUE FOR 1 THREAD!\n")
            exit(99)
    else:
        if x_ind != 0 and x_ind != (dimension_length - 1) and y_ind != 0 and \
                y_ind != (dimension_length-1):  # mid thread
            # output = output[n:-n]

            if DEBUG:
                sys.stderr.write(f"------")
                sys.stderr.write(f"Middle")
                sys.stderr.write(f"------\n")

            offset = [1, -1]
            POSITION = "m"
            #output = get_2d_case_output(offset, output, num_elements_row, "m")
        if x_ind == 0 and y_ind != 0 and y_ind != (dimension_length - 1): #
            # wall on left
            #output = output

            if DEBUG:
                sys.stderr.write(f"------")
                sys.stderr.write(f"Wall Left")
                sys.stderr.write(f"------\n")

            offset = [0, -1]
            POSITION = "m"
            #get_2d_case_output(offset, output, num_elements_row, "m")
        if x_ind == (dimension_length - 1) and y_ind != 0 and y_ind != (
                dimension_length - 1): # wall on right
            #output = output[:-n]

            if DEBUG:
                sys.stderr.write(f"------")
                sys.stderr.write(f"Wall Right")
                sys.stderr.write(f"------\n")

            offset = [1, 0]
            POSITION = "m"
            #get_2d_case_output(offset, output, num_elements_row, "m")
        if y_ind == 0 and x_ind != 0 and x_ind != (dimension_length - 1): #
            # wall on bottom
            #output = output[n:]

            if DEBUG:
                sys.stderr.write(f"------")
                sys.stderr.write(f"Wall Bottom")
                sys.stderr.write(f"------\n")

            offset = [1, -1]
            POSITION = "b"
            #get_2d_case_output(offset, output, num_elements_row, "b")
        if y_ind == (dimension_length - 1) and x_ind != 0 \
                and x_ind != (dimension_length - 1): # wall on top
            #output = output[n:]

            if DEBUG:
                sys.stderr.write(f"------")
                sys.stderr.write(f"Wall Top")
                sys.stderr.write(f"------\n")

            offset = [1, -1]
            POSITION = "t"
            #get_2d_case_output(offset, output, num_elements_row, "t")
        if y_ind == (dimension_length - 1) and x_ind == (dimension_length - 1):
            # top right corner
            #output = output[n:]

            if DEBUG:
                sys.stderr.write(f"------")
                sys.stderr.write(f"Top Right Corner")
                sys.stderr.write(f"------\n")

            offset = [1, 0]
            POSITION = "t"
            #get_2d_case_output(offset, output, num_elements_row, "t")
        if y_ind == 0 and x_ind == (dimension_length - 1): # bottom right corner
            #output = output[n:]

            if DEBUG:
                sys.stderr.write(f"------")
                sys.stderr.write(f"Bottom Right Corner")
                sys.stderr.write(f"------\n")

            offset = [1, 0]
            POSITION = "b"
            #get_2d_case_output(offset, output, num_elements_row, "b")
        if y_ind == 0 and x_ind == 0: # bottom left corner
            #output = output[n:]

            if DEBUG:
                sys.stderr.write(f"------")
                sys.stderr.write(f"Bottom Left Corner")
                sys.stderr.write(f"------\n")

            offset = [0, -1]
            POSITION = "b"
            #get_2d_case_output(offset, output, num_elements_row, "b")
        if y_ind == (dimension_length - 1) and x_ind == 0: # top left corner
            #output = output[n:]

            if DEBUG:
                sys.stderr.write(f"------")
                sys.stderr.write(f"Top Left Corner")
                sys.stderr.write(f"------\n")

            offset = [0, -1]
            POSITION = "t"
            #get_2d_case_output(offset, output, num_elements_row, "t")

    #sys.stderr.write(f"start: {start}\n")
    #sys.stderr.write(f"end: {end}\n")
    #sys.stderr.write(f"output: {output}\n")
    #sys.stderr.write(f"output shape: {output.shape}\n")

    if nt != 1:
        output = get_2d_case_output(offset, output, num_elements_row, POSITION)

    # Task:
    # Set the output array
    # fpp_num = array((end - start, ))
    # print(f"shape fpp_num: {fpp_num.shape}")

    final_num_elements_row = n // dimension_length

    if DEBUG:
        sys.stderr.write(f"final num elements row: {final_num_elements_row}\n")
        sys.stderr.write(f"output length: {len(output)}\n")
        sys.stderr.write(f"output: {output}\n")

    for row_output, row_fpp in enumerate(range(start_y, end_y)):
        if DEBUG:
            sys.stderr.write(f"row_output: {row_output}, row_fpp: {row_fpp}\n")

        start_output = row_output * final_num_elements_row
        end_output = (row_output + 1) * final_num_elements_row

        if DEBUG:
            sys.stderr.write(f"start output: {start_output}\n")
            sys.stderr.write(f"end output: {end_output}\n")

        row_fpp_final = row_fpp * n

        if DEBUG:
            sys.stderr.write(f"row_fpp_final: {row_fpp_final}\n")

        start_fpp = row_fpp_final + start_x
        end_fpp = row_fpp_final + end_x

        if DEBUG:
            sys.stderr.write(f"fpp start: {start_fpp}\n")
            sys.stderr.write(f"fpp end: {end_fpp}\n")

        fpp_num[start_fpp:end_fpp] \
            = output[start_output:end_output]

    #sys.stderr.write(f"(after function call) fpp_num: {fpp_num}\n")

    # need to change...
    #fpp_num[(start*n):(end*n)] = output

def fcn(x, y):
    '''
    This is the function we are studying
    '''
    return cos((x + 1) ** (1. / 3.) + (y + 1) ** (1. / 3.)) + sin(
        (x + 1) ** (1. / 3.) + (y + 1) ** (1. / 3.))


def fcnpp(x, y):
    '''
    This is the second derivative of the function we are studying
    '''
    # Task:
    # Fill this function in with the correct second derivative.  You end up with terms like
    # -cos((x+1)**(1./3.) + (y+1)**(1./3.))*(1./9.)*(x+1)**(-4./3)
    inner = (x + 1) ** (1. / 3.) + (y + 1) ** (1. / 3.)
    # wrt x:
    first_portion_x = (2 * sin(inner) - ((x + 1) ** (1. / 3.)) * cos(inner)) / (
            9 * ((x + 1) ** (5 / 3)))
    second_portion_x = -((x + 1) ** (1 / 3) * sin(inner) + 2 * cos(inner)) / (
            9 * (x + 1) ** (5 / 3))
    # wrt y
    first_portion_y = (2 * sin(inner) - ((y + 1) ** (1 / 3)) * cos(inner)) / (
            9 * (y + 1) ** (5 / 3))
    second_portion_y = -((y + 1) ** (1 / 3) * sin(inner) + 2 * cos(inner)) / (
            9 * (y + 1) ** (5 / 3))
    return first_portion_x + second_portion_x + first_portion_y + second_portion_y

if __name__ == "__main__":
    ##
    # Here are three problem size options for running.  The instructor has chosen these
    # for you.
    option = 2
    if option == 1:
        # Choose this if doing a final run on CARC for your strong scaling study
        NN = array([840 * 6])
        num_threads = [1, 2, 3, 4, 5, 6, 7, 8]
    elif option == 2:
        # Choose this for printing convergence plots on your laptop/lab machine,
        # and for initial runs on CARC.
        # You may want to start with just num_threads=[1] and debug the serial case first.
        NN = 210 * arange(1, 6)
        num_threads = [3]  # eventually include 2, 3
    elif option == 3:
        # Choose this for code development and debugging on your laptop/lab machine
        # You may want to start with just num_threads=[1] and debug the serial case first.
        NN = array([1_000])
        num_threads = [1, 2, 3, 4, 5, 6]  # eventually include 2,3
    else:
        sys.stderr.write("Incorrect Option!")

    ##
    # Begin main computation loop
    ##

    # Task:
    # Initialize your data arrays
    num_threads_length = len(num_threads)
    NN_length = len(NN)  # zeros(int,int)
    error = zeros((num_threads_length,
                   NN_length))  # <array of zeros  (size of num_threads, size of NN) >
    timings = zeros((num_threads_length,
                     NN_length))  # <array of zeros (size of num_threads, size of NN) >

    # personal addition
    #fpp_num = array(())

    # Loop over various numbers of threads
    for i, nt in enumerate(num_threads):
        # Loop over various problem sizes
        for j, n in enumerate(NN):

            # Task:
            # Initialize output array
            fpp_numeric = zeros(
                (n ** 2, ))  # <array of zeros of appropriate size>

            # Task:
            # Choose the number of timings to do for each run
            # changed to 5 from 1
            ntimings = 5  # <insert> ## Check but this seems decent

            # Carry out timing experiment
            min_time = 10000
            for m in range(ntimings):

                # This loop will set up each Thread object to compute fpp numerically in the
                # interior of each thread's domain.  That is, after this loop
                # t_list = [ Thread_object_1, Thread_object_2, ...]
                # where each Thread_object will be ready to compute one thread's contribution
                # to fpp_numeric.  The threads are launched below.
                t_list = []
                for k in range(nt):
                    # Task:
                    # Finish this call to Thread(), passing in the correct target and arguments
                    # args n, nt, k, f, fpp_num
                    t_list.append(Thread(target=compute_fd, args=(n, nt, k, fcn, fpp_numeric)))
                    # t_list.append(Thread(target=<insert>, args=<insert tuple of arguments> ))

                start = timer()
                # Task:
                # Loop over each thread object to launch them.  Then separately loop over each
                # thread object to join the threads.
                for t in t_list:
                    t.start()
                for t in t_list:
                    t.join()
                # < loop over all the threads in t_list and start them >
                # < loop over all the threads in t_list and join them >
                end = timer()
                min_time = min(end - start, min_time)
                print("time", min_time)
            ##
            # End loop over timings
            print(" ")

            ##
            # Use testing-harness to make sure your threaded matvec works
            # This call should print zero (or a numerically zero value)
            if option == 2 or option == 3:
                check_matvec(fpp_numeric, n, fcn)

            # Construct grid of evenly spaced points for a reference evaluation of
            # the double derivative
            h = 1. / (n - 1)
            pts = linspace(0, 1, n)
            X, Y = meshgrid(pts, pts)
            X = X.reshape(-1, )
            Y = Y.reshape(-1, )
            fpp = fcnpp(X, Y)

            # Account for domain boundaries.
            #
            # The boundary_points array is a Boolean array, that acts like a
            # mask on an array.  For example if boundary_points is True at 10
            # points and False at 90 points, then x[boundary_points] will be a
            # length 10 array at those 10 True locations
            boundary_points = (Y == 0)
            fpp_numeric[boundary_points] += (1 / h ** 2) * fcn(X[boundary_points], Y[boundary_points] - h)

            # Task:
            # Account for the domain boundaries at Y == 1, X == 0, X == 1
            boundary_points = (Y == 1)
            fpp_numeric[boundary_points] += (1 / h ** 2) * fcn(X[boundary_points], Y[boundary_points] + h)

            boundary_points = (X == 0)
            fpp_numeric[boundary_points] += (1 / h ** 2) * fcn(X[boundary_points] - h, Y[boundary_points])

            boundary_points = (X == 1)
            fpp_numeric[boundary_points] += (1 / h ** 2) * fcn(X[boundary_points] + h, Y[boundary_points])

            # < include code for these additional boundaries>

            # Task:
            # Compute error
            # sys.stderr.write(f"fpp: {fpp}")
            # sys.stderr.write(f"fpp_numeric: {fpp_numeric}")
            e = abs(fpp - fpp_numeric)
            error[i, j] = L2norm(e, h)
            timings[i, j] = min_time
        # sys.stderr.write(f"error: {error}")

        ##
        # End Loop over various grid-sizes
        print(" ")

        # error[i,:] error for i threads
        # Task:
        # Generate and save plot showing convergence for this thread number
        # --> Comment out plotting before running on CARC
        #quad = linspace()

        #sys.stderr.write(f"error: {error}\n")
        #sys.stderr.write(f"NN: {NN}\n")
        #pyplot.loglog(NN, error[i,:], "r-", ) #<array slice of error values>)
        #pyplot.loglog(NN, 1/(NN**2), "b-")


        # <insert nice formatting options with large axis labels, tick fontsizes, and large legend labels>
        #pyplot.grid(True, which="both", linestyle="dashed")
        #pyplot.title(r"Error Plot for Various Threads")
        #pyplot.legend(["error", "quad. reference"], loc="upper right")
        #pyplot.xlabel(r"$n$")
        #pyplot.ylabel(r"$|e|_{L_2}$")

        #pyplot.savefig('error' + str(i) + '.png', dpi=500, format='png',
        #               bbox_inches='tight', pad_inches=0.0, )
        #pyplot.show()

    strong_scaling_dir = "strong_scaling/"

    strong_scaling_plot = pyplot.figure(1)
    #pyplot.plot(num_threads, timings)  # <array slice of
    # error
    # values>)
    #pyplot.loglog(NN, 1 / (NN ** 2), "bo-")

    # assume there's only one element in NN

    # <insert nice formatting options with large axis labels, tick fontsizes, and large legend labels>
    #pyplot.grid(True, which="both", linestyle="dashed")
    #pyplot.title(f"Strong Scaling, n = {NN[0]}")
    #pyplot.legend(loc="upper left")
    #pyplot.xlabel(f"Threads")
    #pyplot.ylabel(f"Time (s)")

    strong_scaling_filename = f"strong_scaling.png"
    strong_scaling_filepath = strong_scaling_dir + strong_scaling_filename

    pyplot.savefig(strong_scaling_filepath, dpi=500, format='png',
                   bbox_inches='tight', pad_inches=0.0, )
    #pyplot.show()

    strong_scaling_efficiency_plot = pyplot.figure(2)

    # need to fix shape of timings
    timings = timings.ravel()

    # efficiency = np.zeros((num_threads_length, ))
    t_1 = timings[0] # assume 1 thread was the first calculated
    #print(f"t_1: {t_1}")
    #print(f"timings: {timings}")
    #print(f"num_threads: {num_threads}")

    denominator = multiply(timings, num_threads)

    #print(f"denominator: {denominator}")

    #efficiency = t_1 / denominator

    #print(f"efficiency: {efficiency}")

    #pyplot.plot(num_threads, efficiency)  # <array slice of
    # error
    # values>)
    # pyplot.loglog(NN, 1 / (NN ** 2), "bo-")

    # <insert nice formatting options with large axis labels, tick fontsizes, and large legend labels>
    # pyplot.grid(True, which="both", linestyle="dashed")
    #pyplot.title(f"Strong Scaling Efficiency, n = {NN[0]}")
    # pyplot.legend(loc="upper left")
    #pyplot.xlabel(f"Threads")
    #pyplot.ylabel(f"Efficiency")

    #strong_scaling_efficiency_filename = f"strong_scaling_efficiency.png"
    #strong_scaling_efficiency_filepath = strong_scaling_dir + strong_scaling_efficiency_filename

    #pyplot.savefig(strong_scaling_efficiency_filepath, dpi=500, format='png', bbox_inches='tight', pad_inches=0.0, )
    #pyplot.show()






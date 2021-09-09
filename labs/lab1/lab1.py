# Calling an `import *' for a package like numpy imports all the 
# default functions inside numpy, but not all of numpy
from numpy import *
from matplotlib import pyplot

# Task 1: look over these basic plotting commands, watch for x and y to 
#  be printed to the screen

x = linspace(-1,1,120)
print(x)
y = exp(1/2-sin(5*pi*x))
print(y)

pyplot.figure(1)
pyplot.plot(x,y)
# Your figure is saved to a file below with savefig to lab1.png



# Task 2: Try these options to make figure more readable, one at a time
# As you update your figure, with new options, view it.  For instance, on a mac type 
#    $ open -a preview lab1.png

# pyplot.tick_params(labelsize='large')

# pyplot.xlabel(r'$X$', fontsize='large')

# Now, enter the right command for the ylabel using pyplot.ylabel(...)

# Reduce the number of xticks with "pyplot.xticks()"  
# Use ">>> pyplot.xticks? " from inside of iPython to understand the interface 

# Add a parameter to the legend command to make the fontsize large
# pyplot.legend(['First Dataset']) 

pyplot.savefig('lab1.png', dpi=500, format='png', bbox_inches='tight', pad_inches=0.0,)


# Task 3: Try pyplot.show() instead of pyplot.savefig()


# Task 4: To understand loglog plots better, here are the commands from the
# last lecture.  Try making the loglog plot more readable, using the commands from above

#pyplot.figure(2)
#h = 2**-linspace(0,10,1000) 
#pyplot.loglog(h, h, h, h**2, h, h**3) 
#pyplot.show()


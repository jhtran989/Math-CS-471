from numpy import *
from os import system

# Task 1:  Look over this function
def f(x, d):
    if d == 0:
        # return function
        return x**2 - 1
    elif d == 1:
        # return derivative of function
        return 2*x
    elif d == 2:
        # return value of root (for computing error)
        return 1

# Task 2: Look over this Newton's method code 
x = 2.0
s = "Approx. root  " + str(x)[0:7]
print(s)
data = zeros((10,2))

for i in range(10):
    xold = x
    x = x - f(x, 0)/f(x, 1)
    eold = f(xold, 2) - xold
    e = abs(f(x, 2) - x)
    
    s = "Approx. root  " + str(x)[0:7]
    print(s)

    data[i,0] = x
    data[i,1] = e


# Task 3:
# - Your TASK is to dump data to a file and then import it into Latex
# - Look up the "savetxt()" function online, or do this with ">>> savetxt?" inside of ipython, 
#   after doing "from numpy import *".  This function lets you dump data to a text file.
# - When using savetxt(), use the following parameters:  '  &  ' as the delimiter, ' \\\\\n' as 
#   the new line, and  '%.5e' as the ' fmt '.  This will give you a data.tex file similar to meals.tex.  
# - Then, import this data file as a new table into lab1.tex.  Use the importation of meals.tex
#   to guide what you do.  Label and describe this table.

# Task 4: 
# - Add new columns to the data array and to your Latex table that show the convergence 
#   speed of Newton's method 

# Extra if using pdflatex
# - Use the code "system('pdflatex lab1.tex')" to automatically recompile your Latex 
# - Try changing your function f(), or the number of Newton iterations and re-running 
#   this python file.  Re-examining the PDF each time
# - You can even use system('open -a preview lab1.pdf') to open your PDF for you!




from matplotlib import pyplot as plt
import numpy as np

x = np.linspace(-1,1,300)

plt.figure(1)
plt.title("Function 1")
plt.plot(x, x**2)
plt.xlabel(r'$X$', fontsize='large')
plt.ylabel(r'$Y$', fontsize='large')
plt.grid(True)
plt.legend(['$x^2$'])

plt.figure(2)
plt.title("Function 2")
plt.plot(x, x)
plt.xlabel(r'$X$', fontsize='large')
plt.ylabel(r'$Y$', fontsize='large')
plt.grid(True)
plt.legend(['$x$'])



plt.figure(3)
plt.title("Function 3")
plt.plot(x, np.sin(x)+np.cos(x**2))
plt.xlabel(r'$X$', fontsize='large')
plt.ylabel(r'$Y$', fontsize='large')
plt.grid(True)
plt.legend(['$sin(x) + cos(x^2)$'])



plt.show()



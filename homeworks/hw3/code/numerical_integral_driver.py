import numpy
from numpy import exp, cos, pi

import matplotlib.pyplot as plt

from trapezoid_rule import trapezoid_error
from gauss_quadrature_rule import gauss_quadrature_error


def e_cos_function(x, k):
    """
    Acts like a lambda function that evaluates the function exp(cos(k * x))
    at the given parameters x and k (where k should be either pi or pi^2 for
    our example)
    """
    # maybe set k explicitly

    return exp(cos(k * x))

    # if d == 0:
    #     # return f(x) = x^2
    #     return exp(cos(k * x))
    # elif d == 1:
    #     # return the derivative of f(x) = x^2
    #     return 2*x


if __name__ == '__main__':
    # to change the code below, there are two separate things that could be
    # outputted -- one figure for the trapezoid rule (both k values) and
    # another for gauss quadrature (also both k values)
    # the values for n_bound and tol are set to variables first before
    # passing them to the respective functions (see *_rule.py for each of the
    # two numerical methods)
    # currently, the gauss quadrature figure will be saved (can be changed so
    # that both are shown on the same figure -- uncomment the trapezoid parts
    # below)

    # Exact integral:
    # 2.532131755504017 (k = pi)
    # 2.452283895096694 (k = pi^2)

    interval = [-1, 1]

    # both values of k are evaluated (stored as a list)
    k = [pi, pi ** 2]

    n_bound = 250
    tol = 1e-10  # changed from None

    print(f"Numerical integration for exp(cos(k * x))")
    print(f"Interval: [{interval[0]}, {interval[1]}]")

    print(f"--------------------------------")
    print(f"Trapezoid rule (k = pi)")
    print(f"tolerance: {tol}")
    print(f"n bound: {n_bound if tol is None else None}")
    print(f"--------------------------------")

    trapezoid_pi_n_values, trapezoid_pi_delta_values = trapezoid_error(
        e_cos_function, k[0], interval, tol=tol, n_bound=n_bound)

    tol = 1e-7

    print(f"--------------------------------")
    print(f"Trapezoid rule (k = pi^2)")
    print(f"tolerance: {tol}")
    print(f"n bound: {n_bound if tol is None else None}")
    print(f"--------------------------------")

    trapezoid_pi2_n_values, trapezoid_pi2_delta_values = trapezoid_error(
        e_cos_function, k[1], interval, tol=tol, n_bound=100)

    tol = 1e-10  # changed from None

    print(f"--------------------------------")
    print(f"Gauss quadrature rule (k = pi)")
    print(f"tolerance: {tol}")
    print(f"n bound: {n_bound if tol is None else None}")
    print(f"--------------------------------")

    gauss_quadrature_pi_n_values, gauss_quadrature_pi_delta_values = \
        gauss_quadrature_error(e_cos_function, k[0], tol=tol, n_bound=250)

    print(f"--------------------------------")
    print(f"Gauss quadrature rule (k = pi^2)")
    print(f"tolerance: {tol}")
    print(f"n bound: {n_bound if tol is None else None}")
    print(f"--------------------------------")

    gauss_quadrature_pi2_n_values, gauss_quadrature_pi2_delta_values = \
        gauss_quadrature_error(e_cos_function, k[1], tol=tol, n_bound=250)

    # plt.loglog(trapezoid_pi_n_values, trapezoid_pi_delta_values, color='b',
    #            marker='o', label=r'trap, k=$\pi$')
    # plt.loglog(trapezoid_pi2_n_values, trapezoid_pi2_delta_values, color='g',
    #            marker='o', label=r'trap, k=$\pi^2$')

    plt.loglog(gauss_quadrature_pi_n_values,
               gauss_quadrature_pi_delta_values, color='r',
               marker='o', label=r'gauss, k=$\pi$')
    plt.loglog(gauss_quadrature_pi2_n_values,
               gauss_quadrature_pi2_delta_values, color='c',
               marker='o', label=r'gauss, k=$\pi^2$')

    x_reference_values = range(2, 1001)
    linear_reference_values = [x ** -1 for x in x_reference_values]
    quadratic_reference_values = [x ** -2 for x in x_reference_values]
    cubic_reference_values = [x ** -3 for x in x_reference_values]
    exponential_reference_values = [exp(-2 * x) for x in x_reference_values]

    plt.loglog(x_reference_values, linear_reference_values, linestyle="dashed",
               label=r'$O(h)$')
    plt.loglog(x_reference_values, quadratic_reference_values,
               linestyle="dashed",
               label=r'$O(h^2)$')
    plt.loglog(x_reference_values, cubic_reference_values, linestyle="dashed",
               label=r'$O(h^3)$')
    plt.loglog(x_reference_values, exponential_reference_values,
               linestyle="dashed",
               label=r'$O(e^h)$')

    plt.ylim((1e-15, 1e1))

    # plt.title(f"$\\int_{interval[0]}^{interval[0]} e^{a} dx$")
    plt.grid(True, which="both", linestyle="dashed")
    plt.title(r"Numerical integration for $\int_{-1}^{1} e^{cos(kx)} dx$")
    plt.legend(loc="lower left")
    plt.xlabel("n")
    plt.ylabel("error")

    #plt.savefig("numerical_integration_error_plot.png")
    plt.savefig("numerical_integration_error_gaus.png")

    plt.show()

from lglnodes import lglnodes
import numpy as np


# (nodes, weights) = lglnodes(3)
# Compute approximate integral by evaluating your function at
# nodes, then multiply by weights, and finally sum

def gauss_quadrature(function, k, n):
    # x_values, h = np.linspace(interval[0], interval[1], n + 1, retstep=True)
    (nodes, weights) = lglnodes(n)

    approximate_integral = np.multiply(weights, function(nodes, k))
    approximate_integral = sum(approximate_integral)

    return approximate_integral


def gauss_quadrature_error(function, k, tol=None, n_bound=100):
    # Exact integral:
    # 2.532131755504017 (k = pi)
    # 2.452283895096694 (k = pi^2)

    # FIXME
    # print(f"spliced interval: {interval[1:-1]}\n")
    # print(f"{exp(interval)}")

    n = 2
    n_values = []

    # tol = 1e-7

    integral_values = []
    delta_integral_values = []

    # previous_integral = 0
    # current_integral = 0

    delta_integral = 1  # some default value for delta
    # current_k = k[0]
    index = 0

    if tol is None:
        while n < n_bound:
            integral_values.append(gauss_quadrature(function, k,
                                             n))
            # print(f"Approximation of integral [{interval[0]}, "
            #       f"{interval[1]}] of e^cos(pi * x): "
            #       f"{integral_values[index]}\n")

            if index > 0:
                n_values.append(n)
                delta_integral_values.append(abs(integral_values[index] -
                                                 integral_values[index - 1]))

                # print(f"delta_I_{n} = I_{n} - I_{n - 1}:"
                #       f" {delta_integral_values[index - 1]}")

            n += 1
            index += 1
    else:
        while delta_integral > tol:
            integral_values.append(gauss_quadrature(function, k,
                                             n))
            # print(f"Approximation of integral [{interval[0]}, "
            #       f"{interval[1]}] of e^cos(pi * x): "
            #       f"{integral_values[index]}\n")

            if index > 0:
                n_values.append(n)
                delta_integral_values.append(abs(integral_values[index] -
                                                 integral_values[index - 1]))
                delta_integral = delta_integral_values[index - 1]

                # print(f"delta_I_{n} = I_{n} - I_{n - 1}:"
                #       f" {delta_integral_values[index - 1]}")

            n += 1
            index += 1

    # total number of iterations = n_bound - 2 + 1 (does NOT calculate
    # integral at n = n_bound)
    print(f"Total number of iterations: {index}")
    print(f"Final integral approximation: {integral_values[-1]}")
    print(f"Final error (delta_I_{n - 1}): {delta_integral_values[-1]}")
    print()

    return n_values, delta_integral_values

if __name__ == '__main__':
    print("something")
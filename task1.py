#!/opt/software/anaconda/python-3.10.9/bin/python

# =============================================================================
# MIT License
#
# Copyright (c) 2026 Nicholas Young
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================

"""
Version: Python 3.10.9

Description: Using the over-relaxation method to deterministically
			 solve the poisson equation.

Date: 17/03/2026

Author: Nicholas Young
"""

import numpy as np

def solve_poisson_sor(N=101, omega=None, tol=1e-8, max_iter=50000):
    """
    Solve delta phi = f on a 1 m x 1 m square using over-relaxation method.
    """

    # SET GRID (N = No. of grids, h = lattice spacing)
    h = 1.0 / (N - 1)

    # POTENTIAL GIRD
    phi = np.zeros((N, N), dtype=float)

    # SOURCE GRID 'F'(PHI=F)
    f = np.zeros((N, N), dtype=float)

    # BOUNDARY CONDITIONS FOR NxN GRID (VOLTS)
    phi[0, :]  = 100.0 #V
    phi[-1, :] = 100.0 #V
    phi[:, 0]  = 100.0 #V
    phi[:, -1] = 100.0 #V

    # DEFINE OPTIMAL OMEGA (FROM NOTES)
    omega = 2.0 / (1.0 + np.sin(np.pi / N))

    # SOR CALCULATION LOOP
    for iteration in range(max_iter):
        max_change = 0.0

        for i in range(1, N - 1):
            for j in range(1, N - 1):

                # Finite-difference Poisson update
                phi_star = 0.25 * (
                    phi[i+1, j] +
                    phi[i-1, j] +
                    phi[i, j+1] +
                    phi[i, j-1] +
                    h**2 * f[i, j]
                )

                # OVER-RELAXATION STEP
                new_value = (1 - omega) * phi[i, j] + omega * phi_star

                change = abs(new_value - phi[i, j])
                if change > max_change:
                    max_change = change

                phi[i, j] = new_value

        if max_change < tol:
            print(f"Converged in {iteration+1} iterations")
            return phi, h

    print("Did not converge within max_iter")
    return phi, h


# SOLVE
phi, h = solve_poisson_sor()

# VOLTAGE OF THREE RANDOM POINTS TO DETERMINE
points = [(0.50, 0.50), (0.02, 0.02), (0.02, 0.50)]

# PRINT VOLTAGES
for x, y in points:
    i = round(x / h)
    j = round(y / h)
    print(f"phi({x:.2f}, {y:.2f}) = {phi[i, j]:.6f} V")

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

import csv
import math
from pathlib import Path

import numpy as np


def apply_boundary_conditions(phi, boundary_func):
    """Set boundary values on phi in-place."""
    n = phi.shape[0]

    # bottom and top
    for ix in range(n):
        phi[0, ix] = boundary_func(ix, 0, n)         # bottom
        phi[n - 1, ix] = boundary_func(ix, n - 1, n) # top

    # left and right
    for iy in range(n):
        phi[iy, 0] = boundary_func(0, iy, n)         # left
        phi[iy, n - 1] = boundary_func(n - 1, iy, n) # right


def solve_poisson_sor(boundary_func, f, n=101, tol=1e-10, max_iter=50000):
    """
    Solve ∇²phi = f on a 1 m x 1 m square using SOR.

    Array convention:
      phi[iy, ix]
    where iy = row index from bottom to top in the physical interpretation.
    """
    h = 1.0 / (n - 1)

    phi = np.zeros((n, n), dtype=float)
    apply_boundary_conditions(phi, boundary_func)

    omega = 2.0 / (1.0 + np.sin(np.pi / n))

    for iteration in range(max_iter):
        max_change = 0.0

        for iy in range(1, n - 1):
            for ix in range(1, n - 1):
                phi_star = 0.25 * (
                    phi[iy + 1, ix] +
                    phi[iy - 1, ix] +
                    phi[iy, ix + 1] +
                    phi[iy, ix - 1] -
                    h * h * f[iy, ix]
                )

                new_value = (1.0 - omega) * phi[iy, ix] + omega * phi_star
                change = abs(new_value - phi[iy, ix])
                if change > max_change:
                    max_change = change

                phi[iy, ix] = new_value

        # Re-apply boundaries just to be safe
        apply_boundary_conditions(phi, boundary_func)

        if max_change < tol:
            return phi, h, iteration + 1

    return phi, h, max_iter


def point_value(phi, x, y, h):
    """Read phi at physical point (x,y)."""
    ix = int(round(x / h))
    iy = int(round(y / h))
    return phi[iy, ix]


# ----------------------------
# Boundary cases from Task 4
# ----------------------------

def bc_all_100(ix, iy, n):
    return 100.0


def bc_tb_100_lr_minus100(ix, iy, n):
    if iy == 0 or iy == n - 1:
        return 100.0
    if ix == 0 or ix == n - 1:
        return -100.0
    raise ValueError("Not a boundary site")


def bc_tl_200_b_0_r_minus400(ix, iy, n):
    # Use same corner convention as your Task 4 evaluator
    if iy == n - 1:   # top
        return 200.0
    if ix == 0:       # left
        return 200.0
    if iy == 0:       # bottom
        return 0.0
    if ix == n - 1:   # right
        return -400.0
    raise ValueError("Not a boundary site")


# ----------------------------
# Charge cases from Task 4
# ----------------------------

def charge_zero(n, h):
    return np.zeros((n, n), dtype=float)


def charge_uniform_10(n, h):
    return np.full((n, n), 10.0, dtype=float)


def charge_gradient_top1_bottom0(n, h):
    f = np.zeros((n, n), dtype=float)
    for iy in range(n):
        y = iy * h
        f[iy, :] = y
    return f


def charge_exp_centered(n, h):
    f = np.zeros((n, n), dtype=float)
    for iy in range(n):
        y = iy * h
        for ix in range(n):
            x = ix * h
            r = math.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
            f[iy, ix] = math.exp(-10.0 * r)
    return f


def main():
    n = 101
    h = 1.0 / (n - 1)

    points = [
        (0.50, 0.50),
        (0.02, 0.02),
        (0.02, 0.50),
    ]

    boundary_cases = [
        ("BC1_all_edges_+100V", bc_all_100),
        ("BC2_top_bottom_+100_left_right_-100", bc_tb_100_lr_minus100),
        ("BC3_top_left_+200_bottom_0_right_-400", bc_tl_200_b_0_r_minus400),
    ]

    charge_cases = [
        ("Q0_zero_charge", charge_zero(n, h)),
        ("Q1_uniform_10", charge_uniform_10(n, h)),
        ("Q2_gradient_top1_bottom0", charge_gradient_top1_bottom0(n, h)),
        ("Q3_exp_minus10r_centered", charge_exp_centered(n, h)),
    ]

    rows = []

    for bc_name, bc_func in boundary_cases:
        for q_name, f in charge_cases:
            phi, h, n_iter = solve_poisson_sor(bc_func, f, n=n)

            for x, y in points:
                value = point_value(phi, x, y, h)
                rows.append({
                    "point_x_m": x,
                    "point_y_m": y,
                    "boundary_case": bc_name,
                    "charge_case": q_name,
                    "phi_det_V": value,
                    "iterations": n_iter,
                })

                print(
                    f"{bc_name}, {q_name}, ({x:.2f}, {y:.2f}) -> "
                    f"{value:.6f} V   [{n_iter} iterations]"
                )

    out = Path("task5_deterministic_results.csv")
    with out.open("w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(
            fcsv,
            fieldnames=[
                "point_x_m",
                "point_y_m",
                "boundary_case",
                "charge_case",
                "phi_det_V",
                "iterations",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved deterministic results to {out}")


if __name__ == "__main__":
    main()

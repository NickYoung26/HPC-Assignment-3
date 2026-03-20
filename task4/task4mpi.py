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

Description: Using the random walkers to solve poisson equations.

Date: 17/03/2026

Author: Nicholas Young

Task 4 evaluator for PH510 Assignment 3.

Loads Green's-function data from greens_task3_data.npz and evaluates the
potential at the three Task 3 points for the Task 4 boundary conditions
and charge distributions. This uses MPI instead however.

Expected arrays in the .npz file:
    n            : int
    h            : float
    points_xy    : shape (3, 2)
    g_boundary   : shape (n_points, nb)
    se_boundary  : shape (n_points, nb)
    g_charge     : shape (n_points, n, n)
    se_charge    : shape (n_points, n, n)

where nb = 4 * (n - 1).

This script prints a results table and also saves a CSV file.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from mpi4py import MPI


# ---------------------------------------------------------------------
# Boundary indexing
# ---------------------------------------------------------------------

def inverse_boundary_index(b: int, n: int) -> tuple[int, int]:
    """
    Inverse of the boundary indexing used in the Green's-function script.

    Ordering:
      0 .. n-1       : bottom edge, left -> right
      n .. 2n-2      : right edge, bottom+1 -> top
      2n-1 .. 3n-3   : top edge, right-1 -> left
      3n-2 .. 4n-5   : left edge, top-1 -> bottom+1
    """
    if b < n:
        return b, 0  # bottom

    b -= n
    if b < n - 1:
        return n - 1, b + 1  # right

    b -= (n - 1)
    if b < n - 1:
        return n - 2 - b, n - 1  # top

    b -= (n - 1)
    return 0, n - 2 - b  # left


# ---------------------------------------------------------------------
# Task 4 boundary conditions
# ---------------------------------------------------------------------

def bc_all_100(ix: int, iy: int, n: int) -> float:
    """Task 4(a): all edges +100 V."""
    return 100.0


def bc_tb_100_lr_minus100(ix: int, iy: int, n: int) -> float:
    """Task 4(b): top/bottom +100 V, left/right -100 V."""
    if iy == 0 or iy == n - 1:
        return 100.0
    if ix == 0 or ix == n - 1:
        return -100.0
    raise ValueError("Point is not on boundary.")


def bc_tl_200_b_0_r_minus400(ix: int, iy: int, n: int) -> float:
    """Task 4(c): top/left +200 V, bottom 0 V, right -400 V."""
    # Same corner convention as before: top, then left, then bottom, then right
    if iy == n - 1:   # top
        return 200.0
    if ix == 0:       # left
        return 200.0
    if iy == 0:       # bottom
        return 0.0
    if ix == n - 1:   # right
        return -400.0
    raise ValueError("Point is not on boundary.")


def build_boundary_vector(n: int, boundary_func) -> np.ndarray:
    """Precompute boundary values in the same ordering as g_boundary."""
    nb = 4 * (n - 1)
    boundary_values = np.zeros(nb, dtype=float)
    for b in range(nb):
        ix, iy = inverse_boundary_index(b, n)
        boundary_values[b] = boundary_func(ix, iy, n)
    return boundary_values


# ---------------------------------------------------------------------
# Task 4 charge distributions
# ---------------------------------------------------------------------

def charge_zero(n: int, h: float) -> np.ndarray:
    """Zero charge inside the grid."""
    return np.zeros((n, n), dtype=float)


def charge_uniform_10(n: int, h: float) -> np.ndarray:
    """Uniform 10 C over the whole grid."""
    return np.full((n, n), 10.0, dtype=float)


def charge_gradient_top1_bottom0(n: int, h: float) -> np.ndarray:
    """
    Uniform charge gradient from top to bottom.
    Top edge: 1 C m^-2
    Bottom edge: 0 C m^-2
    """
    f = np.zeros((n, n), dtype=float)
    for ix in range(n):
        for iy in range(n):
            y = iy * h
            f[ix, iy] = y
    return f


def charge_exp_centered(n: int, h: float) -> np.ndarray:
    """
    Exponentially decaying charge distribution exp(-10 |r|)
    centred at (0.5, 0.5).
    """
    f = np.zeros((n, n), dtype=float)
    for ix in range(n):
        x = ix * h
        for iy in range(n):
            y = iy * h
            r = math.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
            f[ix, iy] = math.exp(-10.0 * r)
    return f


# ---------------------------------------------------------------------
# Potential evaluation
# ---------------------------------------------------------------------

def phi_from_boundary(
    g_boundary: np.ndarray,
    se_boundary: np.ndarray,
    boundary_values: np.ndarray,
) -> tuple[float, float]:
    """
    Boundary contribution to phi and propagated uncertainty.
    sigma^2(phi_B) = sum_b (sigma(G_b) * V_b)^2
    """
    phi_b = float(np.dot(g_boundary, boundary_values))
    err2_b = float(np.sum((se_boundary * boundary_values) ** 2))
    return phi_b, math.sqrt(err2_b)


def phi_from_charge(
    g_charge: np.ndarray,
    se_charge: np.ndarray,
    f: np.ndarray,
) -> tuple[float, float]:
    """
    Charge contribution to phi and propagated uncertainty.
    sigma^2(phi_f) = sum_{p,q} (sigma(G_charge[p,q]) * f[p,q])^2
    """
    phi_f = float(np.sum(g_charge * f))
    err2_f = float(np.sum((se_charge * f) ** 2))
    return phi_f, math.sqrt(err2_f)


def evaluate_case(
    g_boundary: np.ndarray,
    se_boundary: np.ndarray,
    g_charge: np.ndarray,
    se_charge: np.ndarray,
    boundary_values: np.ndarray,
    f: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    """
    Full potential for one point and one case.
    Returns:
        phi_total, phi_boundary, phi_charge,
        err_total, err_boundary, err_charge
    """
    phi_b, err_b = phi_from_boundary(g_boundary, se_boundary, boundary_values)
    phi_f, err_f = phi_from_charge(g_charge, se_charge, f)
    phi_total = phi_b + phi_f
    err_total = math.sqrt(err_b ** 2 + err_f ** 2)
    return phi_total, phi_b, phi_f, err_total, err_b, err_f


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    npz_path = Path("greens_task3_data.npz")
    if not npz_path.exists():
        raise FileNotFoundError(
            "Could not find greens_task3_data.npz in the current directory."
        )

    data = np.load(npz_path)

    n = int(data["n"])
    h = float(data["h"])
    points_xy = data["points_xy"]
    g_boundary_all = data["g_boundary"]
    se_boundary_all = data["se_boundary"]
    g_charge_all = data["g_charge"]
    se_charge_all = data["se_charge"]

    boundary_cases = [
        ("BC1_all_edges_+100V", build_boundary_vector(n, bc_all_100)),
        ("BC2_top_bottom_+100_left_right_-100", build_boundary_vector(n, bc_tb_100_lr_minus100)),
        ("BC3_top_left_+200_bottom_0_right_-400", build_boundary_vector(n, bc_tl_200_b_0_r_minus400)),
    ]

    charge_cases = [
        ("Q0_zero_charge", charge_zero(n, h)),
        ("Q1_uniform_10", charge_uniform_10(n, h)),
        ("Q2_gradient_top1_bottom0", charge_gradient_top1_bottom0(n, h)),
        ("Q3_exp_minus10r_centered", charge_exp_centered(n, h)),
    ]

    # Build all tasks
    tasks = []
    for p, (x, y) in enumerate(points_xy):
        for bc_name, boundary_values in boundary_cases:
            for q_name, f in charge_cases:
                tasks.append((p, float(x), float(y), bc_name, boundary_values, q_name, f))

    # Static distribution of tasks
    my_tasks = tasks[rank::size]

    local_rows = []

    for p, x, y, bc_name, boundary_values, q_name, f in my_tasks:
        phi_total, phi_b, phi_f, err_total, err_b, err_f = evaluate_case(
            g_boundary_all[p],
            se_boundary_all[p],
            g_charge_all[p],
            se_charge_all[p],
            boundary_values,
            f,
        )

        local_rows.append({
            "point_x_m": x,
            "point_y_m": y,
            "boundary_case": bc_name,
            "charge_case": q_name,
            "phi_total_V": phi_total,
            "phi_boundary_V": phi_b,
            "phi_charge_V": phi_f,
            "stderr_total_V": err_total,
            "stderr_boundary_V": err_b,
            "stderr_charge_V": err_f,
        })

    # Gather results on rank 0
    gathered_rows = comm.gather(local_rows, root=0)

    if rank == 0:
        rows = []
        for chunk in gathered_rows:
            rows.extend(chunk)

        rows.sort(key=lambda r: (
            r["point_x_m"],
            r["point_y_m"],
            r["boundary_case"],
            r["charge_case"],
        ))

        print(f"Loaded Green's-function data from: {npz_path}")
        print(f"N = {n}, h = {h:.6f} m")
        print(f"MPI ranks used = {size}")
        print()

        current_point = None
        for row in rows:
            point = (row["point_x_m"], row["point_y_m"])
            if point != current_point:
                current_point = point
                print(f"Point ({point[0]:.2f} m, {point[1]:.2f} m)")
                print("-" * 95)
                print(
                    f"{'Boundary case':38s} {'Charge case':28s} "
                    f"{'phi_total (V)':>14s} {'stderr (V)':>12s}"
                )

            print(
                f"{row['boundary_case']:38s} {row['charge_case']:28s} "
                f"{row['phi_total_V']:14.6f} {row['stderr_total_V']:12.6f}"
            )

        print()

        csv_path = Path("task4_resultsMPI.csv")
        with csv_path.open("w", newline="", encoding="utf-8") as fcsv:
            writer = csv.DictWriter(
                fcsv,
                fieldnames=[
                    "point_x_m",
                    "point_y_m",
                    "boundary_case",
                    "charge_case",
                    "phi_total_V",
                    "phi_boundary_V",
                    "phi_charge_V",
                    "stderr_total_V",
                    "stderr_boundary_V",
                    "stderr_charge_V",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        print(f"Saved results to {csv_path}")


if __name__ == "__main__":
    main()

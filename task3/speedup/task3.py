#!/opt/software/anaconda/python-3.10.9/bin/python

"""
Green's-function evaluation for PH510 Assignment 3.

This script computes, for selected starting points on a 2D square grid:
  1) The Laplace/edge-potential Green's function:
       G_Laplace(start ; boundary site)
     estimated as the probability that a random walker first exits at each
     boundary site.

  2) The charge-related Green's function:
       G_charge[p, q](start) = h^2 * E[number of visits to site (p, q)]
     estimated from visit counts accumulated over many random walks.

MPI is used to distribute walkers across ranks.

Outputs:
  - .npz data file with Green's functions and simple error estimates
  - one boundary Green's function plot per start point
  - one charge Green's function heatmap per start point
"""

from __future__ import annotations

import os
import time
from typing import List, Tuple

from mpi4py import MPI
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def point_to_index(x_m: float, y_m: float, h: float, n: int) -> Tuple[int, int]:
    """
    Convert physical coordinates (x,y) in metres to array indices (ix, iy),
    where:
      ix runs left -> right
      iy runs bottom -> top
    """
    ix = int(round(x_m / h))
    iy = int(round(y_m / h))

    if not (0 <= ix < n and 0 <= iy < n):
        raise ValueError(f"Point ({x_m}, {y_m}) lies outside the grid.")

    return ix, iy


def is_boundary(ix: int, iy: int, n: int) -> bool:
    """Return True if the site is on the boundary."""
    return ix == 0 or ix == n - 1 or iy == 0 or iy == n - 1


def boundary_linear_index(ix: int, iy: int, n: int) -> int:
    """
    Map a boundary site to a unique 1D boundary index.

    Ordering:
      0 .. n-1                     : bottom edge, left -> right
      n .. 2n-2                   : right edge, bottom+1 -> top
      2n-1 .. 3n-3                : top edge, right-1 -> left
      3n-2 .. 4n-5                : left edge, top-1 -> bottom+1

    Total number of unique boundary sites:
      nb = 4*(n-1)
    """
    if iy == 0:  # bottom, includes both bottom corners
        return ix

    if ix == n - 1:  # right, exclude bottom-right already counted
        return n + (iy - 1)

    if iy == n - 1:  # top, exclude top-right already counted
        return (2 * n - 1) + (n - 2 - ix)

    if ix == 0:  # left, exclude top-left and bottom-left already counted
        return (3 * n - 2) + (n - 2 - iy)

    raise ValueError("Point is not on the boundary.")


def boundary_labels(n: int) -> Tuple[np.ndarray, List[str]]:
    """
    Return x positions and a few edge labels for plotting.
    """
    nb = 4 * (n - 1)
    x = np.arange(nb)
    labels = [
        "bottom start",
        "right start",
        "top start",
        "left start",
    ]
    return x, labels


# ---------------------------------------------------------------------------
# Random walks
# ---------------------------------------------------------------------------

def one_walk(ix0: int, iy0: int, n: int, rng: np.random.Generator) -> Tuple[int, int, np.ndarray]:
    """
    Perform one nearest-neighbour random walk until the boundary is hit.

    Returns
    -------
    hit_ix, hit_iy : int
        Boundary site where the walker first exits.
    visits : np.ndarray, shape (n, n)
        visits[ix, iy] = number of visits to that site during the walk.
        Boundary hit site is not included as a visit count.
    """
    ix, iy = ix0, iy0
    visits = np.zeros((n, n), dtype=np.int32)

    while not is_boundary(ix, iy, n):
        visits[ix, iy] += 1

        step = rng.integers(4)
        if step == 0:
            ix += 1   # right
        elif step == 1:
            ix -= 1   # left
        elif step == 2:
            iy += 1   # up
        else:
            iy -= 1   # down

    return ix, iy, visits

def run_walks_for_point(ix0, iy0, n, h, n_walks, rng):

    """
    Run many walkers for one starting point on one MPI rank.

    Returns
    -------
    local_hits : (nb,) int64
        Boundary-hit counts.
    local_hits_sq : (nb,) int64
        Since hit indicators are 0/1 per walk, storing counts is enough, but
        we keep this shape for consistency: sum of squares == sum for indicators.
    local_visits : (n, n) int64
        Sum of visit counts over all walkers.
    local_visits_sq : (n, n) int64
        Sum of squared visit counts over all walkers.
    n_walks : int
        Number of walkers run locally.
    """

    nb = 4 * (n - 1)

    local_hits = np.zeros(nb, dtype=np.int64)
    local_hits_sq = np.zeros(nb, dtype=np.int64)

    local_visits = np.zeros((n, n), dtype=np.int64)
    local_visits_sq = np.zeros((n, n), dtype=np.int64)

    local_phi_sum = 0.0
    local_phi_sum_sq = 0.0

    for _ in range(n_walks):
        hit_ix, hit_iy, visits = one_walk(ix0, iy0, n, rng)

        bidx = boundary_linear_index(hit_ix, hit_iy, n)
        local_hits[bidx] += 1
        local_hits_sq[bidx] += 1

        local_visits += visits
        local_visits_sq += visits.astype(np.int64) ** 2

        v = boundary_value(hit_ix, hit_iy, n)
        local_phi_sum += v
        local_phi_sum_sq += v * v

    return (
        local_hits,
        local_hits_sq,
        local_visits,
        local_visits_sq,
        local_phi_sum,
        local_phi_sum_sq,
        n_walks,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_boundary_green(
    g_boundary: np.ndarray,
    se_boundary: np.ndarray,
    start_label: str,
    n: int,
    filename: str,
) -> None:
    """Plot edge-potential Green's function."""
    x, _ = boundary_labels(n)

    plt.figure(figsize=(9, 4.5))
    plt.plot(x, g_boundary, linewidth=1.2, label="G_Laplace")
    plt.fill_between(
        x,
        np.maximum(0.0, g_boundary - se_boundary),
        g_boundary + se_boundary,
        alpha=0.25,
        label="±1 s.e."
    )

    # Edge separators
    edge_breaks = [0, n - 1, 2 * n - 1, 3 * n - 2, 4 * (n - 1)]
    for xb in edge_breaks:
        plt.axvline(x=xb, linestyle="--", linewidth=0.8)

    plt.xlabel("Boundary site index")
    plt.ylabel("Probability")
    plt.title(f"Edge-potential Green's function at start {start_label}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=220)
    plt.close()


def plot_charge_green(
    g_charge: np.ndarray,
    start_label: str,
    filename: str,
) -> None:
    """Plot charge-related Green's function as a heatmap."""
    plt.figure(figsize=(6, 5.4))
    plt.imshow(
        g_charge.T,
        origin="lower",
        extent=[0.0, 1.0, 0.0, 1.0],
        aspect="equal",
    )
    plt.colorbar(label="G_charge")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(f"Charge-related Green's function at start {start_label}")
    plt.tight_layout()
    plt.savefig(filename, dpi=220)
    plt.close()

# ---------------------------------------------------------------------------
# PHI CALCULATIONS
# ---------------------------------------------------------------------------

def boundary_value(ix, iy, n):

    """
    Boundary potential at site (ix, iy).
    """

    if iy == 0: 
        return 100.0
    if iy == n - 1: 
        return 0.0
    if ix == 0: 
        return 100.0
    if ix == n - 1: 
        return 0.0

def compute_phi_from_boundary(g_boundary, n):
    nb = 4 * (n - 1)
    phi = 0.0

    for b in range(nb):
        # convert boundary index back to (ix, iy)
        ix, iy = inverse_boundary_index(b, n)
        phi += g_boundary[b] * boundary_value(ix, iy, n)

    return phi

def inverse_boundary_index(b, n):
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

# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main() -> None:
    """
    MPI driver to compute Green's functions for the three required points.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    t0 = time.perf_counter()

    # Grid
    n = 101
    h = 1.0 / (n - 1)

    # TASK 2 and 3 points in one go 
    points_xy = [
        (0.50, 0.50),
        (0.02, 0.02),
        (0.02, 0.50),
    ]
    point_labels = [
        "(0.50 m, 0.50 m)",
        "(0.02 m, 0.02 m)",
        "(0.02 m, 0.50 m)",
    ]

    point_indices = [point_to_index(x, y, h, n) for x, y in points_xy]

    # Monte Carlo settings
    walkers_per_point = 100000

    # Split walkers evenly across ranks
    base = walkers_per_point // size
    extra = walkers_per_point % size
    my_walks = base + (1 if rank < extra else 0)

    if rank == 0:
        print(f"MPI ranks              : {size}")
        print(f"Grid size N            : {n}")
        print(f"Grid spacing h         : {h:.6f} m")
        print(f"Walkers per point      : {walkers_per_point}")
        print(f"Walkers on this run/rank split around : {base} or {base+1}")
        print()

    n_points = len(points_xy)
    nb = 4 * (n - 1)

    # Global containers on root
    if rank == 0:
        g_boundary_all = np.zeros((n_points, nb), dtype=np.float64)
        se_boundary_all = np.zeros((n_points, nb), dtype=np.float64)
        g_charge_all = np.zeros((n_points, n, n), dtype=np.float64)
        se_charge_all = np.zeros((n_points, n, n), dtype=np.float64)
    else:
        g_boundary_all = None
        se_boundary_all = None
        g_charge_all = None
        se_charge_all = None

    for p, ((ix0, iy0), label) in enumerate(zip(point_indices, point_labels)):
        rng = np.random.default_rng(seed=1234567 + 1000 * p + rank)

        (
            local_hits,
            local_hits_sq,
            local_visits,
            local_visits_sq,
            local_phi_sum,
            local_phi_sum_sq,
            local_n,
        ) = run_walks_for_point(ix0, iy0, n, h, my_walks, rng)

        global_hits = np.zeros_like(local_hits)
        global_hits_sq = np.zeros_like(local_hits_sq)
        global_visits = np.zeros_like(local_visits)
        global_visits_sq = np.zeros_like(local_visits_sq)
        global_n = np.array(0, dtype=np.int64)
        global_phi_sum = np.array(0.0, dtype=np.float64)
        global_phi_sum_sq = np.array(0.0, dtype=np.float64)

        comm.Reduce(local_hits, global_hits, op=MPI.SUM, root=0)
        comm.Reduce(local_hits_sq, global_hits_sq, op=MPI.SUM, root=0)
        comm.Reduce(local_visits, global_visits, op=MPI.SUM, root=0)
        comm.Reduce(local_visits_sq, global_visits_sq, op=MPI.SUM, root=0)
        comm.Reduce(np.array(local_n, dtype=np.int64), global_n, op=MPI.SUM, root=0)
        comm.Reduce(np.array(local_phi_sum, dtype=np.float64), global_phi_sum, op=MPI.SUM, root=0)
        comm.Reduce(np.array(local_phi_sum_sq, dtype=np.float64), global_phi_sum_sq, op=MPI.SUM, root=0)

        if rank == 0:
            total_walks = int(global_n)

            # Boundary Green's function = exit probabilities
            g_boundary = global_hits / total_walks

            # For Bernoulli indicator I_b, Var(mean) = p(1-p)/N
            var_boundary_mean = g_boundary * (1.0 - g_boundary) / total_walks
            se_boundary = np.sqrt(np.maximum(0.0, var_boundary_mean))

            # Charge Green's function = h^2 * mean(visits)
            mean_visits = global_visits / total_walks
            mean_visits_sq = global_visits_sq / total_walks
            var_visits = np.maximum(0.0, mean_visits_sq - mean_visits ** 2)
            se_mean_visits = np.sqrt(var_visits / total_walks)

            g_charge = (h ** 2) * mean_visits
            se_charge = (h ** 2) * se_mean_visits

            g_boundary_all[p] = g_boundary
            se_boundary_all[p] = se_boundary
            g_charge_all[p] = g_charge
            se_charge_all[p] = se_charge

            # Values at start point
            g_at_start = g_charge[ix0, iy0]
            se_at_start = se_charge[ix0, iy0]
            

            # Phi Estimate and errors

            phi_est = compute_phi_from_boundary(g_boundary, n)
            mean = global_phi_sum / total_walks
            mean_sq = global_phi_sum_sq / total_walks
            variance = max(0.0, mean_sq - mean * mean)
            stddev = np.sqrt(variance)
            stderr = np.sqrt(variance / total_walks)

            # Greens Summary
            print(f"Start point {label}")
            print(f"  walkers used                     : {total_walks}")
            print(f"  sum G_Laplace over boundary      : {np.sum(g_boundary):.8f}") # sanity check, should = 1
            print(f"  max G_Laplace                    : {np.max(g_boundary):.8e}")
            print(f"  mean s.e.(G_Laplace)             : {np.mean(se_boundary):.8e}")
            print(f"  max s.e.(G_Laplace)              : {np.max(se_boundary):.8e}")
            print()
            print(f"  sum G_charge over all sites      : {np.sum(g_charge):.8e}")
            print(f"  max G_charge                     : {np.max(g_charge):.8e}") # charge of point check
            print(f"  mean s.e.(G_charge)              : {np.mean(se_charge):.8e}") # mean error across grid
            print(f"  max s.e.(G_charge)               : {np.max(se_charge):.8e}") # largest error in grid
            print()
            print(f"  G_charge at start point          : {g_at_start:.8e}")
            print(f"  s.e.(G_charge at start point)    : {se_at_start:.8e}")
            print()

            # Phi Value and Error
            print(f"  Monte Carlo phi                  : {phi_est:.6f} V") # Phi Value
            print(f"  phi from walk average            : {mean:.6f} V") # Should agree with above
            print(f"  stddev(phi samples)              : {stddev:.6f} V")
            print(f"  stderr(phi mean)                 : {stderr:.6f} V")
            print()

            # Begin plotting functions from greens function at point and boundary.

            safe_label = label.replace("(", "").replace(")", "").replace(",", "").replace(" ", "_")

            plot_boundary_green(
                g_boundary,
                se_boundary,
                label,
                n,
                f"greens_boundary_{safe_label}.png",
            )

            plot_charge_green(
                g_charge,
                label,
                f"greens_charge_{safe_label}.png",
            )

    if rank == 0:
        np.savez_compressed(
            "greens_task3_data.npz",
            n=n,
            h=h,
            points_xy=np.array(points_xy, dtype=np.float64),
            g_boundary=g_boundary_all,
            se_boundary=se_boundary_all,
            g_charge=g_charge_all,
            se_charge=se_charge_all,
        )

        elapsed = time.perf_counter() - t0
        print(f"Saved data to greens_task3_data.npz")
        print(f"Total wall time: {elapsed:.3f} s")


if __name__ == "__main__":
    main()


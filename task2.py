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
"""

from mpi4py import MPI  # pylint: disable=import-error
import numpy as np

def point_to_index(x_m, y_m, h):
    """Convert physical coordinates in metres to nearest grid indices."""
    i = int(round(x_m / h))
    j = int(round(y_m / h))
    return i, j


def boundary_value(i, j, n):
    """
    Return the boundary potential when a walker hits the edge.
    """
    if j == 0:         # bottom
        return 100.0
    if j == n - 1:     # top
        return 0
    if i == 0:         # left
        return 100.0
    if i == n - 1:     # right
        return 0

    raise ValueError("Point is not on the boundary.")


def one_walk(i0, j0, n, rng):
    """
    Perform one nearest-neighbour random walk until the boundary is hit.
    Returns the boundary potential at the hit point.
    """
    i, j = i0, j0

    while 0 < i < n - 1 and 0 < j < n - 1:
        step = rng.integers(4)

        if step == 0:
            i += 1   # right
        elif step == 1:
            i -= 1   # left
        elif step == 2:
            j += 1   # up
        else:
            j -= 1   # down

    return boundary_value(i, j, n)


def run_chunk(i0, j0, n, n_walks, rng):
    """
    Run a chunk of walkers from one start point.
    Returns:
        local_sum, local_sum_sq, local_count
    """
    local_sum = 0.0
    local_sum_sq = 0.0

    for _ in range(n_walks):
        v = one_walk(i0, j0, n, rng)
        local_sum += v
        local_sum_sq += v * v

    return local_sum, local_sum_sq, n_walks


def build_tasks(point_indices, walkers_per_point, chunk_size):
    """
    Build a list of tasks.
    Each task is:
        (point_id, i0, j0, walkers_in_chunk)
    """
    tasks = []

    for point_id, (i0, j0) in enumerate(point_indices):
        remaining = walkers_per_point
        while remaining > 0:
            this_chunk = min(chunk_size, remaining)
            tasks.append((point_id, i0, j0, this_chunk))
            remaining -= this_chunk

    return tasks


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Problem setup
    n = 101
    h = 1.0 / (n - 1)

    # Assignment points in metres
    points_xy = [
        (0.50, 0.50),
        (0.02, 0.02),
        (0.02, 0.50),
        (0.98, 0.98)
    ]

    point_indices = [point_to_index(x, y, h) for x, y in points_xy]

    # Monte Carlo settings
    walkers_per_point = 100000
    #chunk_size = walkers_per_point // (4 * size) # POTENTIALLY INEFFICIENT
    chunk_size = 2500

    # Build all tasks on all ranks identically
    tasks = build_tasks(point_indices, walkers_per_point, chunk_size)

    # Static balanced distribution:
    # rank 0 gets tasks 0, size, 2*size, ...
    # rank 1 gets tasks 1, size+1, ...
    my_tasks = tasks[rank::size]

    # Independent RNG per rank
    rng = np.random.default_rng(seed=123456 + rank)

    # Local accumulators per point
    n_points = len(points_xy)
    local_sum = np.zeros(n_points, dtype=np.float64)
    local_sum_sq = np.zeros(n_points, dtype=np.float64)
    local_count = np.zeros(n_points, dtype=np.int64)

    for point_id, i0, j0, n_walks in my_tasks:
        s, s2, c = run_chunk(i0, j0, n, n_walks, rng)
        local_sum[point_id] += s
        local_sum_sq[point_id] += s2
        local_count[point_id] += c

    # Reduce to rank 0
    global_sum = np.zeros(n_points, dtype=np.float64)
    global_sum_sq = np.zeros(n_points, dtype=np.float64)
    global_count = np.zeros(n_points, dtype=np.int64)

    comm.Reduce(local_sum, global_sum, op=MPI.SUM, root=0)
    comm.Reduce(local_sum_sq, global_sum_sq, op=MPI.SUM, root=0)
    comm.Reduce(local_count, global_count, op=MPI.SUM, root=0)

    if rank == 0:
        print(f"Used {size} MPI ranks")
        print(f"Grid size N = {n}, h = {h:.5f} m")
        print(f"Walkers per point = {walkers_per_point}")
        print(f"Chunk size = {chunk_size}")
        print()

        for p, (x, y) in enumerate(points_xy):
            mean = global_sum[p] / global_count[p]
            mean_sq = global_sum_sq[p] / global_count[p]
            variance = max(0.0, mean_sq - mean * mean)
            stderr = np.sqrt(variance / global_count[p])

            print(
                f"Point ({x:.2f} m, {y:.2f} m): "
                f"phi ≈ {mean:.6f} V, "
                f"standard error ≈ {stderr:.6f} V, "
                f"walkers = {global_count[p]}"
            )


if __name__ == "__main__":
    main()



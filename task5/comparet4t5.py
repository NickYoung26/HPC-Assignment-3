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

Compare stochastic (Green's-function) and deterministic (SOR) results
for PH510 Assignment 3 Task 5.

Expected input files:
  - task4_results.csv
  - task5_deterministic_results.csv

Output:
  - task5_comparison.csv

Checks whether the stochastic and deterministic values agree within the
Monte Carlo statistical error bars.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path


def to_float(row, key):
    return float(row[key])


def make_key(row):
    """
    Match rows by point and case labels.
    Rounded point coordinates avoid tiny string-format mismatches.
    """
    return (
        round(float(row["point_x_m"]), 8),
        round(float(row["point_y_m"]), 8),
        row["boundary_case"],
        row["charge_case"],
    )


def main():
    stochastic_path = Path("task4_results.csv")
    deterministic_path = Path("task5_deterministic_results.csv")
    output_path = Path("task5_comparison.csv")

    if not stochastic_path.exists():
        raise FileNotFoundError(f"Missing file: {stochastic_path}")
    if not deterministic_path.exists():
        raise FileNotFoundError(f"Missing file: {deterministic_path}")

    # Load stochastic results
    with stochastic_path.open("r", newline="", encoding="utf-8") as f:
        stochastic_rows = list(csv.DictReader(f))

    # Load deterministic results
    with deterministic_path.open("r", newline="", encoding="utf-8") as f:
        deterministic_rows = list(csv.DictReader(f))

    det_map = {make_key(row): row for row in deterministic_rows}

    comparison_rows = []

    print("Task 5 comparison")
    print("-" * 120)
    print(
        f"{'Point':18s} {'Boundary case':38s} {'Charge case':28s} "
        f"{'phi_MC':>12s} {'sigma_MC':>12s} {'phi_det':>12s} "
        f"{'diff':>12s} {'z':>10s} {'within 1σ':>10s}"
    )

    for srow in stochastic_rows:
        key = make_key(srow)

        if key not in det_map:
            print(f"Warning: no deterministic match for key {key}")
            continue

        drow = det_map[key]

        x = to_float(srow, "point_x_m")
        y = to_float(srow, "point_y_m")

        phi_mc = to_float(srow, "phi_total_V")
        sigma_mc = to_float(srow, "stderr_total_V")
        phi_det = to_float(drow, "phi_det_V")

        diff = phi_mc - phi_det

        if sigma_mc > 0.0:
            z_score = diff / sigma_mc
            within_1sigma = abs(z_score) <= 1.0
            within_2sigma = abs(z_score) <= 2.0
        else:
            z_score = float("nan")
            within_1sigma = abs(diff) < 1e-12
            within_2sigma = within_1sigma

        comparison_rows.append({
            "point_x_m": x,
            "point_y_m": y,
            "boundary_case": srow["boundary_case"],
            "charge_case": srow["charge_case"],
            "phi_stochastic_V": phi_mc,
            "stderr_stochastic_V": sigma_mc,
            "phi_deterministic_V": phi_det,
            "difference_V": diff,
            "abs_difference_V": abs(diff),
            "z_score": z_score,
            "within_1sigma": within_1sigma,
            "within_2sigma": within_2sigma,
            "phi_boundary_V": to_float(srow, "phi_boundary_V"),
            "phi_charge_V": to_float(srow, "phi_charge_V"),
            "stderr_boundary_V": to_float(srow, "stderr_boundary_V"),
            "stderr_charge_V": to_float(srow, "stderr_charge_V"),
        })

        point_label = f"({x:.2f}, {y:.2f})"
        z_str = f"{z_score:.3f}" if math.isfinite(z_score) else "nan"

        print(
            f"{point_label:18s} "
            f"{srow['boundary_case']:38s} "
            f"{srow['charge_case']:28s} "
            f"{phi_mc:12.6f} {sigma_mc:12.6f} {phi_det:12.6f} "
            f"{diff:12.6f} {z_str:>10s} {str(within_1sigma):>10s}"
        )

    # Save CSV
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "point_x_m",
                "point_y_m",
                "boundary_case",
                "charge_case",
                "phi_stochastic_V",
                "stderr_stochastic_V",
                "phi_deterministic_V",
                "difference_V",
                "abs_difference_V",
                "z_score",
                "within_1sigma",
                "within_2sigma",
                "phi_boundary_V",
                "phi_charge_V",
                "stderr_boundary_V",
                "stderr_charge_V",
            ],
        )
        writer.writeheader()
        writer.writerows(comparison_rows)

    print()
    print(f"Saved comparison table to {output_path}")

    # Optional summary
    n_total = len(comparison_rows)
    n_1sigma = sum(1 for row in comparison_rows if row["within_1sigma"])
    n_2sigma = sum(1 for row in comparison_rows if row["within_2sigma"])

    print(f"Within 1σ: {n_1sigma}/{n_total}")
    print(f"Within 2σ: {n_2sigma}/{n_total}")


if __name__ == "__main__":
    main()


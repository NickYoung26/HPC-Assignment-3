# HPC-Assignment-3
The aim of this assignment is investigate methods for solving the Laplace (∇φ = 0) and Poisson (∇φ = f ) equations in two dimensions. Each directory contains the associated files related to that specific task.

It should be noted that any files in the repository not labelled here can be considered irrelevant in completion of the tasks, hence can be ignored.

TASK 1

task1.py - Deterministic solution of the poisson equation.
sbatchT1.sh  - Sbatch script
task1-slurm-7247472.out - Code output. Phi values for points, no. of iterations before convergence.

TASK 2

task2.py - Stochastic solution of the poisson equation
sbatcht2.sh  - Sbatch script
task2-slurm-7247638.out - greens function data for boundary and grid points, errors and phi values (although uneeded here).

TASK 3

task2.py - Stochastic solution of the poisson equation for the 3 points (2cm,2cm), (50cm,50cm), (2cm, 50cm).
sbatcht3.sh  - Sbatch script.
task3-slurm-7247640.out - code output, greens functions points, phi values, errors etc.
greens_task3_data.npz - Greens functions data to use for task 4.

speedup - directory containing simplified task3 code for the speedup calculation.

Greens function charge plots at boundaries for 3 start points:
    greens_boundary_0.02_m_0.02_m.png
    greens_boundary_0.50_m_0.50_m.png
    greens_boundary_0.02_m_0.50_m.png 

Greens function of charge plots at grid points for 3 start points:
    greens_charge_0.02_m_0.02_m.png
    greens_charge_0.50_m_0.50_m.png
    greens_charge_0.02_m_0.50_m.png
  
TASK 4

greens_task3_data.npz - Greens function data found from task 3 which we can apply to determine all the cases here.

task4.py - single core calculation version
task4_results.csv - Table of results for every case required (single core method)

task4mpi.py - mpi 16 core calculation version
task4_resultsMPI.csv - Table of stochastic results for every case considered (16 core method).

sbatcht4 - 1 core Job script
sbatcht4mpi.sh - mpi job script

task4-slurm-7247498.out - timing of single core task 4 calculations.
task4mpi-slurm-7247742.out - timing of 16 core task 4 calculations.

TASK 5

task5.py - Determinstic calculation for each case required
task5_deterministic_results.csv - Table of deterministic results.

task4_results.csv - Table of stochastic results for every case required from task 4.

comparet4t5.py - code to compare task4 results (stochastic) and task 5 (deterministic).
task5_comparisonsimple.csv - table showing comparison of task 4 and task 5 results.

sbatcht5.sh - job script for task5.py and comparet4t5.py

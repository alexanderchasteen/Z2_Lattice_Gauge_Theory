#! /bin/bash
#SBATCH --time=48:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=20
#SBATCH --mem=100GB
#SBATCH --partition=open
#SBATCH -A open
#SBATCH --output=Z2PT_%j.out
#SBATCH --error=Z2PT_%j.err

#---load modules on Roar Collab--
module purge
module load gcc

#--- run---
gcc -O3 -fopenmp Z2_lattice_parallel_tempering.c -lm -o Z2_lattice_parallel_tempering

srun ./Z2_lattice_parallel_tempering


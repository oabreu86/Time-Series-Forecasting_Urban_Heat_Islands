#!/bin/bash

#SBATCH --job-name=landsat
#SBATCH --output=landsat.out
#SBATCH --ntasks=8
#SBATCH --partition=broadwl
#SBATCH --constraint=fdr
#SBATCH --account=macs30123

# Load Python and MPI modules
module load python/anaconda-2019.03
module load intelmpi/2018.2.199+intel-18.0

# Run the python program with mpirun
python3 calc_features_func.py
mpirun python3 get_landsat_data.py


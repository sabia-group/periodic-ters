#!/bin/bash

#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=72
#SBATCH --export=NONE
#SBATCH --time=00:01:00


# prepare environment
module load mkl intel/19.1.2
module load impi/2019.8
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$INTEL_HOME/compilers_and_libraries/linux/lib/intel64:$MKL_HOME/lib/intel64/:$HOME/.local/lib
module list

exe='/u/brek/build/FHIaims/build/aims.250131.scalapack.mpi.x'
ulimit -s unlimited
export OMP_NUM_THREADS=1

# prepare file names
log='aims.out'

# print some information
echo ' started at:' `date`
echo '   hostname:' `hostname`
echo

# run
srun $exe >> $log

# report finish time
echo 'finished at:' `date`

#!/bin/bash
#SBATCH -J matrix_multi_mpi_5        # Use a new job name to be sure
#SBATCH --ntasks=30               
#SBATCH --time=0:10:00             # 1 minute should be plenty
#SBATCH --mail-user=edoardo.frulla@studenti.polito.it
#SBATCH --mail-type=ALL
#SBATCH --partition=cpu_skylake

# --- 1. Environment Setup ---
echo "JOB STARTING"
echo "Purging modules to ensure a clean environment..."
module purge

# --- 2. Load Required Modules ---
echo "Loading gcc and openmpi..."
module load openmpi/4.1.8_gcc11
echo "Loaded modules are:"
module list # This will show us what is loaded

source /home/hpc_2025_group_04/s343236/virtual_python_environment/bin/activate
 
make mpi_mm

# --- 5. Execution ---
# Let's check what Slurm has allocated to us
echo "SLURM_NTASKS allocated: $SLURM_NTASKS"
echo "Running the MPI executable..."

# Use the variables to ensure we run the correct file with the correct number of tasks
mpirun -np $SLURM_NTASKS ./build/mpi_mm.out
#srun ./$EXEC_NAME

echo "JOB FINISHED"

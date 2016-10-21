#!/bin/bash

#SBATCH --time=00:40:00   # walltime
#SBATCH --ntasks=24   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=5G   # memory per CPU core
#SBATCH -J "Free Benchmark"   # job name
#SBATCH -p m8
#SBATCH --mail-user=cojoco@gmail.com   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH -o "/fslhome/cojoco/compute/slurmOutput/classtm/slurm-%j.out"
#SBATCH -e "/fslhome/cojoco/compute/slurmOutput/classtm/slurm-%j.out"

# Compatibility variables for PBS. Delete if not needed.
export PBS_NODEFILE=`/fslapps/fslutils/generate_pbs_nodefile`
export PBS_JOBID=$SLURM_JOB_ID
export PBS_O_WORKDIR="$SLURM_SUBMIT_DIR"
export PBS_QUEUE=batch

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
if [ -z "$1" || -z "$2" || -z "$3" ]
  then
    echo "Usage: sbatch main.slurm numTopics free/log variational/sampling"
    exit 1
fi
settings="${HOME}/classtm/settings/amazonData/$3/$1topics$2.settings"
outputdir="/fslhome/cojoco/compute/classtm/amazonSampling/"
module load python/3/5

. "${HOME}/classtm/ENV/bin/activate"
for i in {1..24}
  do
    python3 submain.py $settings $outputdir $i &
  done
wait

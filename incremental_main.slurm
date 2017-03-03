#!/bin/bash

#SBATCH --time=08:00:00   # walltime
#SBATCH --ntasks=24   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=2560M   # memory per CPU core
#SBATCH -p m8
#SBATCH -J "Newsgroups Benchmark"   # job name
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
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]
  then
    echo "Usage: sbatch main.slurm random_seed min_job_num max_job_num {log,free}"
    exit 1
fi
settings="${HOME}/classtm/settings/incrementalNewsgroups/newsgroupsNumLabeledGraph$4.settings"
outputdir="/fslhome/cojoco/compute/classtm/incNews/numLabels/"
module load python/3/5

. "${HOME}/classtm/ENV/bin/activate"

RANDOM=$1
job_start_num=$2
job_end_num=$3

for ((i=$job_start_num;i<=$job_end_num;i++))
  do
    randomnum=$RANDOM
    echo "random number was $randomnum"
    python3 incremental_submain.py $settings $outputdir $i $randomnum &
  done
wait

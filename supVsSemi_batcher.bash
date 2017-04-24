#! /usr/bin/env bash

job_start_num=1
job_end_num=1
jobs_per_core=1

supprefix="${HOME}/classtm/settings/incrementalNewsgroups/sweepLabeled/newsgroupsSupervisedLong"
semiprefix="${HOME}/classtm/settings/incrementalNewsgroups/sweepLabeled/newsgroupsSemisupervisedLong"
outputdir="/fslhome/cojoco/compute/free/newsSup/"

for ((i=$job_start_num;i<=$job_end_num;i=$i+$jobs_per_core))
  do
    randomnum=$RANDOM
    end_job_num=$(( $i+$jobs_per_core-1 ))
    
    sbatch sup_main.slurm $randomnum $i $end_job_num ${supprefix}log.settings ${outputdir}
    sbatch sup_main.slurm $randomnum $i $end_job_num ${supprefix}free.settings ${outputdir}
    sbatch semi_main.slurm $randomnum $i $end_job_num ${semiprefix}log.settings ${outputdir}
    sbatch semi_main.slurm $randomnum $i $end_job_num ${semiprefix}free.settings ${outputdir}
  done
wait

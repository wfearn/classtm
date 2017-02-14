#! /usr/bin/env bash

job_start_num=1
job_end_num=100
jobs_per_core=5

for ((i=$job_start_num;i<=$job_end_num;i=$i+5))
  do
    randomnum=$RANDOM
    end_job_num=$(( $i+$jobs_per_core-1 ))
    
    sbatch incremental_main.slurm $randomnum $i $end_job_num log
    sbatch incremental_main.slurm $randomnum $i $end_job_num free
  done
wait

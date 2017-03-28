#! /usr/bin/env bash

job_start_num=1
job_end_num=1
jobs_per_core=1

for ((i=$job_start_num;i<=$job_end_num;i=$i+$jobs_per_core))
  do
    randomnum=$RANDOM
    end_job_num=$(( $i+$jobs_per_core-1 ))
    
    sbatch sup_main.slurm $randomnum $i $end_job_num log
    sbatch sup_main.slurm $randomnum $i $end_job_num free
    sbatch semi_main.slurm $randomnum $i $end_job_num log
    sbatch semi_main.slurm $randomnum $i $end_job_num free
  done
wait

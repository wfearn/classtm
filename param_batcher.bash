#! /usr/bin/env bash

job_start_num=1
job_end_num=50
jobs_per_core=1

weights="100%
10%
1
500"
smoothings="0.01
0.0001"
numlabels="100
1000
10000"


for ((i=$job_start_num;i<=$job_end_num;i=$i+$jobs_per_core))
  do
    randomnum=$RANDOM
    end_job_num=$(( $i+$jobs_per_core-1 ))
    for weight in "$weights"
    do
      for smooth in "$smoothings"
      do
        for labels in "$numlabels"
        do
          settingsprefix="${HOME}/classtm/settings/incrementalNewsgroups/"
          outputdirprefix="/fslhome/cojoco/compute/free/newsParams/"
          settings="${settingsprefix}${labels}labeled/${weight}weight${smooth}smoothfree.settings"
          outputdir="${outputdirprefix}${labels}labeled/"
          sbatch param_main.slurm $randomnum $i $end_job_num $settings $outputdir
        done
      done
    done
  done
wait

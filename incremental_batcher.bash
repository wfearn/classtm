#! /usr/bin/env bash

job_start_num=1
job_end_num=100
jobs_per_core=1

settingsprefix="${HOME}/classtm/settings/incrementalNewsgroups/sweepLabeled/newsgroupsNumLabeledGraph"
outputdir="/fslhome/cojoco/compute/free/newsLabels/"

for ((i=$job_start_num;i<=$job_end_num;i=$i+$jobs_per_core))
  do
    randomnum=$RANDOM
    end_job_num=$(( $i+$jobs_per_core-1 ))
    
    sbatch incremental_main.slurm $randomnum $i $end_job_num ${settingsprefix}log.settings $outputdir
    sbatch incremental_main.slurm $randomnum $i $end_job_num ${settingsprefix}free.settings $outputdir
    sbatch incremental_main.slurm $randomnum $i $end_job_num ${settingsprefix}rf.settings $outputdir
    sbatch incremental_main.slurm $randomnum $i $end_job_num ${settingsprefix}nb.settings $outputdir
    sbatch incremental_main.slurm $randomnum $i $end_job_num ${settingsprefix}svm.settings $outputdir
    sbatch incremental_main.slurm $randomnum $i $end_job_num ${settingsprefix}tsvm.settings $outputdir
  done
wait

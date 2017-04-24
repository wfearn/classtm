#! /usr/bin/env bash

job_start_num=41
job_end_num=50
jobs_per_core=1
topic_start_num=20
topic_end_num=500
topic_jump=20

settingsprefix="${HOME}/classtm/settings/incrementalNewsgroups/sweepNumTopics/"
outputdir="/fslhome/cojoco/compute/free/newsTopics/"

for ((i=$job_start_num;i<=$job_end_num;i=i+$jobs_per_core))
  do
    randomnum=$RANDOM
    end_job_num=$(( $i+$jobs_per_core-1 ))
    
    for (( numtopics=$topic_start_num;numtopics<=$topic_end_num;numtopics=$numtopics+$topic_jump ))
    do
      sbatch num_topics_main.slurm $randomnum $i $end_job_num ${settingsprefix}${numtopics}topicsfree.settings $outputdir
      sbatch num_topics_main.slurm $randomnum $i $end_job_num ${settingsprefix}${numtopics}topicslog.settings $outputdir
      sbatch num_topics_main.slurm $randomnum $i $end_job_num ${settingsprefix}${numtopics}topicsrf.settings $outputdir
      sbatch num_topics_main.slurm $randomnum $i $end_job_num ${settingsprefix}${numtopics}topicsnb.settings $outputdir
      sbatch num_topics_main.slurm $randomnum $i $end_job_num ${settingsprefix}${numtopics}topicssvm.settings $outputdir
      sbatch num_topics_main.slurm $randomnum $i $end_job_num ${settingsprefix}${numtopics}topicstsvm.settings $outputdir
    done
  done
wait

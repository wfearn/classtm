#! /usr/bin/env bash

job_start_num=1
job_end_num=1
jobs_per_core=1
topic_start_num=20
topic_end_num=20
topic_jump=20

for ((i=$job_start_num;i<=$job_end_num;i=i+$jobs_per_core))
  do
    randomnum=$RANDOM
    end_job_num=$(( $i+$jobs_per_core-1 ))
    
    for (( numtopics=$topic_start_num;numtopics<=$topic_end_num;numtopics=$numtopics+$topic_jump ))
    do
      sbatch num_topics_main.slurm $numtopics free $randomnum $i $end_job_num
      sbatch num_topics_main.slurm $numtopics log $randomnum $i $end_job_num
      sbatch num_topics_main.slurm $numtopics svm $randomnum $i $end_job_num
      sbatch num_topics_main.slurm $numtopics rf $randomnum $i $end_job_num
      sbatch num_topics_main.slurm $numtopics nb $randomnum $i $end_job_num
      sbatch num_topics_main.slurm $numtopics tsvm $randomnum $i $end_job_num
    done
  done
wait

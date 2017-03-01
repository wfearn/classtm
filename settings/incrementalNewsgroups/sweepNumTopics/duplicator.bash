#!/usr/bin/env bash

# Duplicates a file, changing the number of topics

startnum=20
endnum=500
skipnum=20

for ((i=$startnum;i<=$endnum;i=$i+$skipnum));
do
  sed "s/numtopics   40/numtopics   $i/" newsgroupsSemisupervisedlog.settings > ${i}topicslog.settings
done

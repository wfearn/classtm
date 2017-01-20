#!/usr/bin/env bash
# This adds 'anchors_file  ' + the name of the anchors file to each settings
#   file

for fullname in /fslhome/cojoco/compute/classtm/anchorsNewsgroups/*; do
  filename=$(basename $fullname)
  echo "anchors_file  $fullname" >> ./${filename}topicsfree.settings
  echo "anchors_file  $fullname" >> ./${filename}topicslog.settings
done

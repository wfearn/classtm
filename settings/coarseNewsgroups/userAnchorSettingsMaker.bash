#!/usr/bin/env bash

for fullname in /fslhome/cojoco/compute/classtm/anchorsNewsgroups/*; do
  filename=$(basename $fullname)
  cp /fslhome/cojoco/classtm/settings/coarseNewsgroups/variational/20topicsfree.settings /fslhome/cojoco/classtm/settings/coarseNewsgroups/variational/${filename}topicsfree.settings
  cp /fslhome/cojoco/classtm/settings/coarseNewsgroups/variational/20topicslog.settings /fslhome/cojoco/classtm/settings/coarseNewsgroups/variational/${filename}topicslog.settings
  cp /fslhome/cojoco/classtm/settings/coarseNewsgroups/sampling/20topicsfree.settings /fslhome/cojoco/classtm/settings/coarseNewsgroups/sampling/${filename}topicsfree.settings
  cp /fslhome/cojoco/classtm/settings/coarseNewsgroups/sampling/20topicslog.settings /fslhome/cojoco/classtm/settings/coarseNewsgroups/sampling/${filename}topicslog.settings
done

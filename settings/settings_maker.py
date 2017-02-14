#!/usr/bin/env python3
"""--------INCOMPLETE--------

Makes settings files for experiments
"""

import argparse
import os


def _run(args):
  # parameters to use for the settings files, change these if needed
  # experiment options
  seed='531'
  testsize='1000'
  startlabeled='1000'
  endlabeled='10000'
  increment='1000'

  #model options
  model='incfree'
  numtopics='40'
  expgrad_epsilon='1e-4'
  lda_helper='variational'
  label_weight='500'
  smoothing='0.01'

  # The directory to put settings files into
  directory = args.directory
  # The pickle file to use for this experiment
  pickle_file = args.pickle_file
  # The list of subgroups to make settings files for
  subgroups = args.subgroup
  os.makedirs(directory, exist_ok=True)
  for subgroup in subgroups:
    os.makedirs(os.path.join(directory, subgroup), exist_ok=True)


if __name__ == '__main__':
  subgroup_help = ''.join(['a subgroup of settings files to make, one subdirectory will',
                           'be made for each subgroup'])
  parser = argparse.ArgumentParser()
  parser.add_argument('directory',
                      help='the directory to put the settings files')
  parser.add_argument('pickle_file',
                      help='the pickle file to use in the experiments')
  parser.add_argument('-s', '--subgroup', action='append',
                      help=subgroup_help)
  args = parser.parse_args()
#  _run(args)
print('INCOMPLETE, DO NOT USE')

"""Check whether smoothing has a good effect or not"""
import os
import pickle
import random
import subprocess
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np

from activetm import utils
from classtm import evaluate


FILE_DIR = os.path.dirname(__file__)
REPO_DIR = os.path.join(FILE_DIR, os.pardir)
OUT_DIR = '/local/okuda/tmp'


def _run_experiments(settingses, num):
    """Run experiments with settings in settingses, each repeated num times"""
    for i in range(num):
        cur_seed = random.randint(0, sys.maxsize)
        for settings in settingses:
            print('====', cur_seed, settings, '====', flush=True)
            start = time.time()
            subprocess.run([
                'python3',
                os.path.join(REPO_DIR, 'incremental_submain.py'),
                os.path.join(FILE_DIR, settings),
                OUT_DIR,
                str(i),
                str(cur_seed)])
            print('####', time.time() - start, flush=True)


def _plot_results(settingses, num):
    """Plot results"""
    groupnames = [utils.parse_settings(settings)['group']
                  for settings in settingses]
    acc_data = []
    anchortime_data = []
    applytime_data = []
    traintime_data = []
    for group in groupnames:
        acc_data.append([])
        anchortime_data.append([])
        applytime_data.append([])
        traintime_data.append([])
        groupdir = os.path.join(OUT_DIR, group)
        for i in range(num):
            filename = str(i) + '.results'
            with open(os.path.join(groupdir, filename), 'rb') as ifh:
                cur_results = pickle.load(ifh)
            for result in cur_results:
                acc_data[-1].append(
                    evaluate.accuracy(result['confusion_matrix']))
            # aggregates time data without regard for how many labeled
            # documents there are
            anchortime_data[-1].append(
                result['anchorwords_time'].total_seconds())
            applytime_data[-1].append(
                result['applytrain_time'].total_seconds())
            traintime_data[-1].append(
                result['train_time'].total_seconds())

    name = 'tmp'

    fig, axis = plt.subplots(1, 1)
    bp_dict = axis.boxplot(acc_data, labels=groupnames)
    # http://stackoverflow.com/questions/18861075
    for line in bp_dict['medians']:
        x, y = line.get_xydata()[1]
        plt.text(x, y, '%.4f' % y, horizontalalignment='left',
                 verticalalignment='center')
    fig.savefig(name+'_acc.pdf')

    fig, axis = plt.subplots(1, 1)
    anchormeans = np.mean(anchortime_data, axis=1)
    anchorstd = np.std(anchortime_data, axis=1)
    applymeans = np.mean(applytime_data, axis=1)
    applystd = np.std(applytime_data, axis=1)
    trainmeans = np.mean(traintime_data, axis=1)
    trainstd = np.std(traintime_data, axis=1)
    ind = np.arange(len(groupnames))
    width = 0.35
    p1 = axis.bar(ind, anchormeans, width, yerr=anchorstd)
    p2 = axis.bar(ind, applymeans, width, bottom=anchormeans, yerr=applystd)
    p3 = axis.bar(ind, trainmeans, width, bottom=(anchormeans+applymeans),
                  yerr=trainstd)
    plt.xticks(ind, tuple(groupnames))
    axis.set_ylabel('Time (seconds)')
    for rect in p1:
        height = rect.get_height()
        axis.text(rect.get_x() + rect.get_width(), height,
                '{0:.2f}'.format(height),
                va='center')
    fig.legend((p1[0], p2[0], p3[0]), ('Anchor Words', 'Topic Mixtures',
        'Training'))
    fig.savefig(name+'_time.pdf')


def _run():
    """Run experiments and plot data"""
    settingses = [
        'sup.settings',
        'supnormed.settings',
        'projected.settings',
        'zeroneg.settings',
        ]
    num = 100
    # _run_experiments(settingses, num)
    _plot_results(settingses, num)


if __name__ == '__main__':
    _run()

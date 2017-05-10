"""Check whether smoothing has a good effect or not"""
import os
import pickle
import random
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from activetm import utils
from classtm import evaluate


REPO_DIR = os.path.dirname(__file__)
OUT_DIR = '/local/okuda/tmp'


def _run_experiments(settingses, num):
    """Run experiments with settings in settingses, each repeated num times"""
    for i in range(num):
        cur_seed = random.randint(0, sys.maxsize)
        for settings in settingses:
            subprocess.run([
                'python3',
                os.path.join(REPO_DIR, 'incremental_submain.py'),
                os.path.join(REPO_DIR, settings),
                OUT_DIR,
                str(i),
                str(cur_seed)])


def _plot_results(settingses):
    """Plot results"""
    groupnames = [utils.parse_settings(settings)['group']
                  for settings in settingses]
    data = []
    for group in groupnames:
        data.append([])
        groupdir = os.path.join(OUT_DIR, group)
        for filename in os.listdir(groupdir):
            if filename.endswith('.results'):
                with open(os.path.join(groupdir, filename), 'rb') as ifh:
                    cur_results = pickle.load(ifh)
                for result in cur_results:
                    data[-1].append(
                        evaluate.accuracy(result['confusion_matrix']))

    fig, axis = plt.subplots(1, 1)
    bp_dict = axis.boxplot(data, labels=groupnames)
    # http://stackoverflow.com/questions/18861075
    for line in bp_dict['medians']:
        x, y = line.get_xydata()[1]
        plt.text(x, y, '%.4f' % y, horizontalalignment='left',
                 verticalalignment='center')
    fig.savefig('results.pdf')


def _run():
    """Run experiments and plot data"""
    settingses = [
        'check.settings',
        'check_zero.settings']
    _run_experiments(settingses, 10)
    _plot_results(settingses)


if __name__ == '__main__':
    _run()

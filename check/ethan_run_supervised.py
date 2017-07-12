"""Check whether smoothing has a good effect or not"""
import os
import pickle
import random
import subprocess
import sys
import time

from activetm import utils
from classtm import evaluate


FILE_DIR = os.path.dirname(__file__)
REPO_DIR = os.path.join(FILE_DIR, os.pardir)
OUT_DIR = os.environ.get('CLASSTM_OUT_DIR', '/local/okuda/tmp')


def _run_experiments(settingses, num):
    """Run experiments with settings in settingses, each repeated num times"""
    for i in range(num):
        cur_seed = random.randint(0, sys.maxsize)
        for settings in settingses:
            print('====', cur_seed, settings, '====', flush=True)
            start = time.time()
            subprocess.run([
                'python3',
                os.path.join(REPO_DIR, 'submain.py'),
                os.path.join(FILE_DIR, settings),
                OUT_DIR,
                str(i),
                str(cur_seed)])
            print('####', time.time() - start, flush=True)


def _run():
    """Run experiments and plot data"""
    settingses = [
        'ethan_supervised_sank.settings',
        'ethan_supervised_free_classifier.settings',
        ]
    num = 100
    _run_experiments(settingses, num)


if __name__ == '__main__':
    _run()

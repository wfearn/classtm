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

def _run_supervised_experiments(settingses, num, seed):
    """Run experiments with settings in settingses, each repeated num times"""
    for i in range(num):
        for settings in settingses:
            print('====', seed, settings, '====', flush=True)
            start = time.time()
            subprocess.run([
                'python3',
                os.path.join(REPO_DIR, 'submain.py'),
                os.path.join(FILE_DIR, settings),
                OUT_DIR,
                str(i),
                str(seed)])
            print('####', time.time() - start, flush=True)

def _run_semi_supervised_experiments(settingses, num, seed):
    """Run experiments with settings in settingses, each repeated num times"""
    for i in range(num):
        for settings in settingses:
            print('====', seed, settings, '====', flush=True)
            start = time.time()
            subprocess.run([
                'python3',
                os.path.join(REPO_DIR, 'incremental_submain.py'),
                os.path.join(FILE_DIR, settings),
                OUT_DIR,
                str(i),
                str(seed)])
            print('####', time.time() - start, flush=True)

def _run_all_the_things():
    seed = random.randint(0, sys.maxsize)
    num = 100

    """Run experiments and plot data"""
    supervised_settingses = [
        'ethan_supervised_sank.settings',
        'ethan_supervised_free_classifier.settings',
        ]

    semi_supervised_settingses = [
        'ethan_semisupervised_free_classifier.settings',
        'ethan_semisupervised_sank.settings',
        ]

    _run_supervised_experiments(supervised_settingses, num, seed)
    _run_semi_supervised_experiments(semi_supervised_settingses, num, seed)

if __name__ == '__main__':
    _run_all_the_things()

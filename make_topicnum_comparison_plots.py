import argparse
import numpy as np
import datetime
import getpass
import logging
import os
import pickle
import shutil
import subprocess
import sys
import threading
import time

import ankura
import classtm.models

from classtm import plot
from classtm import labeled
from activetm import utils

'''
The output from an experiment should take the following form:

    settings1
      run1_1
      run2_1
      ...
settings1 can be any name
'''


def get_groups(outputdir):
    runningdir = os.path.join(outputdir, 'running')
    result = [x[0] for x in os.walk(outputdir) if x[0] != runningdir and x[0] != outputdir]
    return sorted(result)


def extract_data(fpath):
    with open(fpath, 'rb') as ifh:
        return pickle.load(ifh)


def get_data(dirname):
    data = []
    for f in os.listdir(dirname):
        if not f.endswith('results'):
            continue
        fpath = os.path.join(dirname, f)
        if os.path.isfile(fpath):
            data.append(extract_data(fpath))
    return data


def get_stats(accs):
    """Gets the stats for a data dict"""
    keys = sorted(accs.keys())
    bot_errs = []
    top_errs = []
    meds = []
    means = []

    for key in keys:
        bot_errs.append(np.mean(accs[key]) - np.percentile(accs[key], 25))
        top_errs.append(np.percentile(accs[key], 75) - np.mean(accs[key]))
        meds.append(np.median(accs[key]))
        means.append(np.mean(accs[key]))
    print('bot_errs', bot_errs)
    print('top_errs', top_errs)
    print('meds', meds)
    print('means', means)
    return {'bot_errs': bot_errs, 'top_errs': top_errs, 'meds': meds, 'means': means}


def get_accuracy(datum):
    """Gets accuracy from the confusion matrix of the datum"""
    correct = 0
    incorrect = 0

    for class1 in datum['confusion_matrix']:
        row = datum['confusion_matrix'][class1]
        for class2 in row:
            if class1 == class2:
                correct += row[class2]
            else:
                incorrect += row[class2]
    accuracy = correct / (correct + incorrect)
    return accuracy


def get_time(datum):
    """Gets time taken to run from the datum"""
    time = datum['train_time'] + datum['applytrain_time'] + datum['anchorwords_time']
    time = time.total_seconds()
    return time


def make_label(xlabel, ylabel):
    """Makes a labels object to pass to make_plot"""
    return {'xlabel': xlabel, 'ylabel': ylabel}


def make_plot(datas, topicnums, labels, outputdir, filename, colors):
    """Makes a single plot with multiple lines
    
    datas: {['free' or 'log']: [float]}
    topicnums: [int]
    labels: {'xlabel': string, 'ylabel': string}
    outputdir: string
    filename: string
    colors: returned by plot.get_separate_colors(int)
    """
    new_plot = plot.Plotter(colors)
    min_y = float('inf')
    max_y = float('-inf')
    for key, data in datas.items():
        stats = get_stats(data)
        for mean in stats['means']:
            for bot_err in stats['bot_errs']:
                min_y = min(min_y, mean - bot_err)
            for top_err in stats['top_errs']:
                max_y = max(max_y, mean + top_err)
        # get the line's name
        name = 'Unknown Classifier'
        if key == 'free':
            name = 'Free Classifier'
        elif key == 'log':
            name = 'Logistic Regression'
        elif key == 'nb':
            name = 'Naive Bayes'
        elif key == 'rf':
            name = 'Random Forest'
        elif key == 'svm':
            name = 'SVM'
        elif key == 'tsvm':
            name = 'Transductive SVM'
        # plot the line
        new_plot.plot(topicnums[key],
                  stats['means'],
                  name,
                  stats['meds'],
                  yerr=[stats['bot_errs'], stats['top_errs']])
    new_plot.set_xlabel(labels['xlabel'])
    new_plot.set_ylabel(labels['ylabel'])
    new_plot.set_ylim([min_y, max_y])
    new_plot.savefig(os.path.join(outputdir, filename))


def make_plots(outputdir, dirs):
    """Makes plots from the data"""
    colors = plot.get_separate_colors(6)
    dirs.sort()

    # get the data from the files
    data = {}
    acc_datas = {}
    time_datas = {}
    for dir in dirs:
        try:
            if dir[-4:] == 'free':
                if 'free' in data.keys():
                    data['free'].extend(get_data(dir))
                else:
                    data['free'] = get_data(dir)
            elif dir[-3:] == 'log':
                if 'log' in data.keys():
                    data['log'].extend(get_data(dir))
                else:
                    data['log'] = get_data(dir)
            elif dir[-2:] == 'nb':
                if 'nb' in data.keys():
                    data['nb'].extend(get_data(dir))
                else:
                    data['nb'] = get_data(dir)
            elif dir[-2:] == 'rf':
                if 'rf' in data.keys():
                    data['rf'].extend(get_data(dir))
                else:
                    data['rf'] = get_data(dir)
#            elif dir[-3:] == 'svm' and dir[-4:] != 'tsvm':
#                if 'svm' in data.keys():
#                    data['svm'].extend(get_data(dir))
#                else:
#                    data['svm'] = get_data(dir)
#            elif dir[-4:] == 'tsvm' and dir[-5:] != '_tsvm':
#                if 'tsvm' in data.keys():
#                    data['tsvm'].extend(get_data(dir))
#                else:
#                    data['tsvm'] = get_data(dir)
            else:
                print('directory', dir, 'not being used in the graph')
        except EOFError:
            print('read empty file in', dir)

    for key, dat in data.items():
        if key not in acc_datas.keys():
            acc_datas[key] = {}
            time_datas[key] = {}
        for datum in dat:
            for subdatum in datum:
                topic_num = subdatum['model'].topics.shape[1]
                if topic_num not in acc_datas[key].keys():
                    acc_datas[key][topic_num] = []
                    time_datas[key][topic_num] = []
                acc_datas[key][topic_num].append(get_accuracy(subdatum))
                time_datas[key][topic_num].append(get_time(subdatum))
    # plot the data
    topicnums = {}
    for key, dat in acc_datas.items():
        topicnums[key] = list(sorted(acc_datas[key].keys()))
    # first plot accuracy
    acc_labels = make_label('Number of Topics', 'Accuracy')
    make_plot(acc_datas, topicnums, acc_labels, outputdir, 'accuracy.pdf', colors)
    # then plot time
    time_labels = make_label('Number of Topics', 'Time to Train')
    make_plot(time_datas, topicnums, time_labels, outputdir, 'times.pdf', colors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launcher for ActiveTM '
            'experiments')
    parser.add_argument('outputdir', help='directory for output')
    args = parser.parse_args()

    try:
        if not os.path.exists(args.outputdir):
            logging.getLogger(__name__).error('Cannot write output to: '+args.outputdir)
            sys.exit(-1)
        groups = get_groups(args.outputdir)
        make_plots(args.outputdir, groups)
    except Exception as e:
        print(e)
        raise

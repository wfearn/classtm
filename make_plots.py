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

    sampling
      settings1free
        run1_1
        run2_1
        ...
      settings2free
      ...
      settings1log
        run1_1
        run2_1
        ...
      settings2log
      ...
    variational
      settings1free
        run1_1
        run2_1
        ...
      settings2free
      ...
      settings1log
        run1_1
        run2_1
        ...
      settings2log
      ...

This is the format the plotter is expecting. There will be at most four lines
plotted, two for sampling (one for the logistic classifier, one for the free
classifier) and two for variational (same as for sampling). The order of free
and log does not matter, it just looks for "free" or "log" at the end of the
directory name.
'''


def generate_settings(filename):
    with open(filename) as ifh:
        for line in ifh:
            line = line.strip()
            if line:
                yield line


def get_hosts(filename):
    hosts = []
    with open(args.hosts) as ifh:
        for line in ifh:
            line = line.strip()
            if line:
                hosts.append(line)
    return hosts


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
    time = datum['eval_time'] + datum['init_time'] + datum['train_time']
    time = time.total_seconds()
    return time


def make_label(xlabel, ylabel):
    """Makes a labels object to pass to make_plot"""
    return {'xlabel': xlabel, 'ylabel': ylabel}


def make_plot(datas, free_var_topics, log_var_topics, free_sam_topics, log_sam_topics, labels, outputdir, filename, colors):
    """Makes a single plot with multiple lines
    
    datas: {['free' or 'log']: [float]}
    num_topics: [int]
    labels: {'xlabel': string, 'ylabel': string}
    outputdir: string
    filename: string
    colors: returned by plot.get_separate_colors(int)
    """
    new_plot = plot.Plotter(colors)
    min_y = float('inf')
    max_y = float('-inf')
    for key, data in datas.items():
        topics = None
        stats = get_stats(data)
        for mean in stats['means']:
            for bot_err in stats['bot_errs']:
                min_y = min(min_y, mean - bot_err)
            for top_err in stats['top_errs']:
                max_y = max(max_y, mean + top_err)
        # get the line's name
        name = 'Unknown Classifier'
        if key == 'free_var':
            topics = free_var_topics
            name = 'Free Classifier w/ Variational'
        elif key == 'log_var':
            topics = log_var_topics
            name = 'Logistic Classifier w/ Variational'
        if key == 'free_sam':
            topics = free_sam_topics
            name = 'Free Classifier w/ Sampling'
        elif key == 'log_sam':
            topics = log_sam_topics
            name = 'Logistic Classifier w/ Sampling'
        # plot the line
        new_plot.plot(topics,
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
    colors = plot.get_separate_colors(4)
    dirs.sort()
    free_var_accuracy = {}
    log_var_accuracy = {}
    free_sam_accuracy = {}
    log_sam_accuracy = {}
    free_var_times = {}
    log_var_times = {}
    free_sam_times = {}
    log_sam_times = {}
    free_var_topics = {}
    log_var_topics = {}
    free_sam_topics = {}
    log_sam_topics = {}
    for d in dirs:
        if 'variational' in d:
            # pull out the data
            data = get_data(os.path.join(outputdir, d))
            for datum in data:
                datum_topicnum = datum['model'].numtopics
                if type(datum['model']) is classtm.models.FreeClassifyingAnchor:
                    free_var_topics[datum_topicnum] = 0
                    if datum_topicnum not in free_var_accuracy:
                        free_var_accuracy[datum_topicnum] = []
                        free_var_times[datum_topicnum] = []
                    free_var_accuracy[datum_topicnum].append(get_accuracy(datum))
                    free_var_times[datum_topicnum].append(get_time(datum))
                elif type(datum['model']) is classtm.models.LogisticAnchor:
                    log_var_topics[datum_topicnum] = 0
                    if datum_topicnum not in log_var_accuracy:
                        log_var_accuracy[datum_topicnum] = []
                        log_var_times[datum_topicnum] = []
                    log_var_accuracy[datum_topicnum].append(get_accuracy(datum))
                    log_var_times[datum_topicnum].append(get_time(datum))
        if 'sampling' in d:
            # pull out the data
            data = get_data(os.path.join(outputdir, d))
            for datum in data:
                datum_topicnum = datum['model'].numtopics
                if type(datum['model']) is classtm.models.FreeClassifyingAnchor:
                    free_sam_topics[datum_topicnum] = 0
                    if datum_topicnum not in free_sam_accuracy:
                        free_sam_accuracy[datum_topicnum] = []
                        free_sam_times[datum_topicnum] = []
                    free_sam_accuracy[datum_topicnum].append(get_accuracy(datum))
                    free_sam_times[datum_topicnum].append(get_time(datum))
                elif type(datum['model']) is classtm.models.LogisticAnchor:
                    log_sam_topics[datum_topicnum] = 0
                    if datum_topicnum not in log_sam_accuracy:
                        log_sam_accuracy[datum_topicnum] = []
                        log_sam_times[datum_topicnum] = []
                    log_sam_accuracy[datum_topicnum].append(get_accuracy(datum))
                    log_sam_times[datum_topicnum].append(get_time(datum))

    # plot the data
    free_var_topics = sorted(free_var_topics.keys())
    log_var_topics = sorted(log_var_topics.keys())
    free_sam_topics = sorted(free_sam_topics.keys())
    log_sam_topics = sorted(log_sam_topics.keys())
    # first plot accuracy
    acc_datas = {}
    if free_var_accuracy:
        acc_datas['free_var'] = free_var_accuracy
    if log_var_accuracy:
        acc_datas['log_var'] = log_var_accuracy
    if free_sam_accuracy:
        acc_datas['free_sam'] = free_sam_accuracy
    if log_sam_accuracy:
        acc_datas['log_sam'] = log_sam_accuracy
    if not acc_datas:
        print('No accuracy data collected! Are you using a new model?')
    acc_labels = make_label('Number of Topics', 'Accuracy')
    make_plot(acc_datas, free_var_topics, log_var_topics, free_sam_topics, log_sam_topics, acc_labels, outputdir, 'accuracy.pdf', colors)
    # then plot time
    time_datas = {}
    if free_var_times:
        time_datas['free_var'] = free_var_times
    if log_var_times:
        time_datas['log_var'] = log_var_times
    if free_sam_times:
        time_datas['free_sam'] = free_sam_times
    if log_sam_times:
        time_datas['log_sam'] = log_sam_times
    if not time_datas:
        print('No time data collected! Are you using a new model?')
    time_labels = make_label('Number of Topics', 'Time to Complete')
    make_plot(time_datas, free_var_topics, log_var_topics, free_sam_topics, log_sam_topics, time_labels, outputdir, 'times.pdf', colors)

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

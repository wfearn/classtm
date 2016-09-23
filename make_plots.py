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

    output_directory
        settings1
            run1_1
            run2_1
            ...
        settings2
        ...

In this way, it gets easier to plot the results, since each settings will make a
line on the plot, and each line will be aggregate data from multiple runs of the
same settings.
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
    true_pos = datum['confusion_matrix']['pos']['pos']
    true_neg = datum['confusion_matrix']['neg']['neg']
    false_pos = datum['confusion_matrix']['neg']['pos']
    false_neg = datum['confusion_matrix']['pos']['neg']

    d_true = 0
    d_false = 0
    d_true += true_pos
    d_true += true_neg
    d_false += false_pos
    d_false += false_neg
    accuracy = d_true / (d_true + d_false)
    return accuracy


def get_time(datum):
    time = datum['eval_time'] + datum['init_time'] + datum['train_time']
    time = time.total_seconds()
    return time


def make_label(xlabel, ylabel):
    return {'xlabel': xlabel, 'ylabel': ylabel}


def make_plot(datas, free_topics, log_topics, labels, outputdir, filename, colors):
    """Makes a single plot with multiple lines
    
    datas: {['free' or 'log']: [float]}
    num_topics: [int]
    labels: {'xlabel': string, 'ylabel': string}
    outputdir: string
    filename: string
    colors: returned by plot.get_separate_colors(int)
    """
    print(datas)
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
        if key == 'free':
            topics = free_topics
            name = 'Free Classifier'
        elif key == 'log':
            topics = log_topics
            name = 'Logistic Classifier'
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
    colors = plot.get_separate_colors(len(dirs))
    dirs.sort()
    free_accuracy = {}
    log_accuracy = {}
    free_times = {}
    log_times = {}
    free_topics = {}
    log_topics = {}
    for d in dirs:
        # pull out the data
        print(d)
        data = get_data(os.path.join(outputdir, d))
        eval_times = []
        init_times = []
        train_times = []
        models = []
        for datum in data:
            datum_topicnum = datum['model'].numtopics
            if type(datum['model']) is classtm.models.FreeClassifyingAnchor:
                free_topics[datum_topicnum] = 0
                if datum_topicnum not in free_accuracy:
                    free_accuracy[datum_topicnum] = []
                    free_times[datum_topicnum] = []
                free_accuracy[datum_topicnum].append(get_accuracy(datum))
                free_times[datum_topicnum].append(get_time(datum))
            elif type(datum['model']) is classtm.models.LogisticAnchor:
                log_topics[datum_topicnum] = 0
                if datum_topicnum not in log_accuracy:
                    log_accuracy[datum_topicnum] = []
                    log_times[datum_topicnum] = []
                log_accuracy[datum_topicnum].append(get_accuracy(datum))
                log_times[datum_topicnum].append(get_time(datum))

    # plot the data
    free_topics = sorted(free_topics.keys())
    log_topics = sorted(log_topics.keys())
    # first plot accuracy
    acc_datas = {}
    if free_accuracy:
        acc_datas['free'] = free_accuracy
    if log_accuracy:
        acc_datas['log'] = log_accuracy
    if not acc_datas:
        print('No accuracy data collected! Are you using a new model?')
    acc_labels = make_label('Number of Topics', 'Accuracy')
    make_plot(acc_datas, free_topics, log_topics, acc_labels, outputdir, 'accuracy.pdf', colors)
    # then plot time
    time_datas = {}
    if free_times:
        time_datas['free'] = free_times
    if log_times:
        time_datas['log'] = log_times
    if not time_datas:
        print('No time data collected! Are you using a new model?')
    time_labels = make_label('Number of Topics', 'Time to Complete')
    make_plot(time_datas, free_topics, log_topics, time_labels, outputdir, 'times.pdf', colors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launcher for ActiveTM '
            'experiments')
    parser.add_argument('working_dir', help='ActiveTM directory '
            'available to hosts (should be a network path)')
    parser.add_argument('outputdir', help='directory for output (should be a '
            'network path)')
    args = parser.parse_args()

    try:
        begin_time = datetime.datetime.now()
        runningdir = os.path.join(args.outputdir, 'running')
        if os.path.exists(runningdir):
            shutil.rmtree(runningdir)
        try:
            os.makedirs(runningdir)
        except OSError:
            pass
        if not os.path.exists(args.outputdir):
            logging.getLogger(__name__).error('Cannot write output to: '+args.outputdir)
            sys.exit(-1)
        groups = get_groups(args.outputdir)
        make_plots(args.outputdir, groups)
        run_time = datetime.datetime.now() - begin_time
        with open(os.path.join(args.outputdir, 'run_time'), 'w') as ofh:
            ofh.write(str(run_time))
        os.rmdir(runningdir)
    except Exception as e:
        print(e)
        raise

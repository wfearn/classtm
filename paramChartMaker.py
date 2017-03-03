#! /usr/bin/env python3
"""
Makes a CSV file from a ClassTM results files when doing hyperparameter testing

The directories should be formatted like the following:

    {numlabels1}labeled
        {weight1}weight-{smoothing1}smoothing
            1.results
            2.results
            3.results
            ...
        {weight2}weight-{smoothing1}smoothing
            1.results
            2.results
            3.results
            ...
        {weight1}weight-{smoothing2}smoothing
            1.results
            2.results
            3.results
            ...
        {weight2}weight-{smoothing2}smoothing
            1.results
            2.results
            3.results
            ...
        ...
    {numlabels2}labeled
        {weight1}weight-{smoothing1}smoothing
            1.results
            2.results
            3.results
            ...
        {weight2}weight-{smoothing1}smoothing
            1.results
            2.results
            3.results
            ...
        {weight1}weight-{smoothing2}smoothing
            1.results
            2.results
            3.results
            ...
        {weight2}weight-{smoothing2}smoothing
            1.results
            2.results
            3.results
            ...
        ...
    ...
"""

import argparse
import csv
import os
import pickle

import ankura
import classtm


def get_accuracy(datum):
    """Gets accuracy from the confusion matrix of the datum"""
    correct = 0
    incorrect = 0

    for class1 in datum[0]['confusion_matrix']:
        row = datum[0]['confusion_matrix'][class1]
        for class2 in row:
            if class1 == class2:
                correct += row[class2]
            else:
                incorrect += row[class2]
    accuracy = correct / (correct + incorrect)
    return accuracy


def extract_data(fpath):
    """Extract data from a pickle file"""
    with open(fpath, 'rb') as ifh:
        return pickle.load(ifh)


def get_data(dirname):
    """Get the data from a directory of results files"""
    data = []
    for f in os.listdir(path=dirname):
        if not f.endswith('results'):
            continue
        fpath = os.path.join(dirname, f)
        if os.path.isfile(fpath):
            data.append(extract_data(fpath))
    return data


def get_average_accuracy(dirname):
    """Gets the average accuracy from a directory of classtm results"""
    data = get_data(dirname)
    accs = []
    for datum in data:
        accs.append(get_accuracy(datum))
    average = sum(accs) / len(accs)
    return average


def make_chart(datadir_container):
    """Make a chart from the files in all of the data dirs in this dir"""
    chart = {}
    for e in os.listdir(path=datadir_container):
        if 'weight' in e and 'smoothing' in e:
            weight, smoothing = e.split('-')
            weight = weight[:-6]
            smoothing = smoothing[:-9]
            avg_acc = get_average_accuracy(os.path.join(datadir_container, e))
            chart[(weight, smoothing)] = avg_acc
        if 'logistic' in e:
            avg_acc = get_average_accuracy(os.path.join(datadir_container, e))
            chart[('logistic', 'nosmooth')] = avg_acc
    return chart


def _run(args):
    basedir = args.directory
    charts = {}
    for d in os.listdir(path=basedir):
        if 'labeled' in d:
            numlabeled = d[:-7]
            charts[numlabeled] = make_chart(os.path.join(basedir, d))
    sorted_numlabeled = sorted(charts.keys())
    for numlabeled in sorted_numlabeled:
        with open(numlabeled + 'labels.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['weight', 'smoothing', 'average accuracy'])
            sorted_weight_smoothing = sorted(charts[numlabeled].keys(), key=lambda el: (el[0], el[1]))
            for weight_smoothing in sorted_weight_smoothing:
                avg_acc = charts[numlabeled][weight_smoothing]
                writer.writerow([weight_smoothing[0], weight_smoothing[1], str(avg_acc)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make a chart from results')
    parser.add_argument('directory', help='Base directory containing \
                        "{numlabeled}labeled" folders')
    args = parser.parse_args()
    _run(args)

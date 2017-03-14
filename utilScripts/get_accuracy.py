#!/usr/bin/env python3

import os
import pickle
import sys

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
    print('accuracy:', accuracy)
    return accuracy


def extract_data(fpath):
    """Extract data from a pickle file"""
    with open(fpath, 'rb') as ifh:
        return pickle.load(ifh)


def get_data(dirname):
    """Get the data from a directory of results files"""
    data = []
    for f in os.listdir(dirname):
        if not f.endswith('results'):
            continue
        fpath = os.path.join(dirname, f)
        if os.path.isfile(fpath):
            data.append(extract_data(fpath))
    return data


def _run(dirname):
    """Run the main program"""
    data = get_data(dirname)
    accs = []
    for datum in data:
        accs.append(get_accuracy(datum))
    average = sum(accs) / len(accs)
    print('average value was:', average)


if __name__ == '__main__':
    _run(sys.argv[1])

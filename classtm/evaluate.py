"""Evaluation for classification"""

import datetime
import time


def confusion_matrix(model, words, labels, classorder):
    """Build confusion matrix for model"""
    result = {}
    for cla in classorder:
        result[cla] = {}
        for inner in classorder:
            result[cla][inner] = 0
    # This gets the time to apply topics to the test documents and predict their values
    start = time.time()
    predictions = model.predict(words)
    end = time.time()
    devtest_time = datetime.timedelta(seconds=end-start)
    for prediction, label in zip(predictions, labels):
        result[label][prediction] += 1
    return result, devtest_time


def accuracy(confusion_matrix):
    """Calculate true positive rate based on confusion_matrix"""
    total = 0
    correct = 0
    for label, preds in confusion_matrix.items():
        for pred, count in preds.items():
            if pred == label:
                correct += count
            total += count
    return correct / total

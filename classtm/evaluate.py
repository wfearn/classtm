"""Evaluation for classification"""


def confusion_matrix(model, words, labels, classorder):
    """Build confusion matrix for model"""
    result = {}
    for cla in classorder:
        result[cla] = {}
        for inner in classorder:
            result[cla][inner] = 0
    for tokens, label in zip(words, labels):
        prediction = model.predict(tokens)
        result[label][prediction] += 1
    return result


"""Prints out topics of trained model"""
import argparse
import pickle


def parse_args():
    """Parses arguments"""
    parser = argparse.ArgumentParser(
        description='Prints out topics of trained model')
    parser.add_argument(
        'results',
        help='path to pickle containing experiment results')
    parser.add_argument(
        'corpus',
        help='path to pickle containing corpus used in experiment')
    return parser.parse_args()


def _get_trained_to_corpus(model):
    """Builds converter from trained vocabulary to corpus vocabulary"""
    converter = {}
    for i, val in enumerate(model.corpus_to_train_vocab):
        converter[val] = i
    return converter


def _get_topic_words(topic, converter, vocab, num):
    """Gets top num words from topic"""
    # sort indices by value contained in cell in ascending order, then take the
    # last num of these sorted indices, and flip the order; thus, topwords is
    # the top num indices of topic
    topwords = topic.argsort()[-num:][::-1]
    return [vocab[converter[a]] for a in topwords]


def _run():
    args = parse_args()
    with open(args.results, 'rb') as ifh:
        results = pickle.load(ifh)
    with open(args.corpus, 'rb') as ifh:
        corpus = pickle.load(ifh)
    model = results[-1]['model']
    converter = _get_trained_to_corpus(model)
    for i in range(model.topics.shape[1]):
        topwords = _get_topic_words(
            model.topics[:, i],
            converter,
            corpus.vocab,
            10)
        print(topwords)


if __name__ == '__main__':
    _run()

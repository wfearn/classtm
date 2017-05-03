"""Runs classification experiment

Assumes that pickled dataset is a(n) AbstractClassifiedDataset
"""
import argparse
import datetime
import os
import pickle
import random
import socket
import time

from activetm import utils
import classtm.models
from classtm import evaluate


def ensure_dir_exists(dirname):
    """Ensure that directory exists"""
    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError:
            pass


def partition_data_ids(num_docs, rng, settings):
    """Parition data"""
    testsize = int(settings['testsize'])
    shuffled_doc_ids = list(range(num_docs))
    rng.shuffle(shuffled_doc_ids)
    return shuffled_doc_ids[:testsize], shuffled_doc_ids[testsize:]


def get_lda_helper(lda_type):
    """Get topic inference algorithm"""
    if lda_type == 'variational':
        return classtm.models.VariationalHelper
    elif lda_type == 'sampling':
        return classtm.models.SamplingHelper
    else:
        raise ValueError("No lda_helper of type " + lda_type)


# pylint:disable-msg=too-many-locals
def _run():
    parser = argparse.ArgumentParser(description='Job runner for ClassTM')
    parser.add_argument(
            'settings',
            help='''the path to a file containing settings, as described in \
            README.md in the root ClassTM directory''')
    parser.add_argument('outputdir', help='directory for output')
    parser.add_argument('label', help='identifying label')
    parser.add_argument('seed', default=-1, type=int, nargs='?')
    args = parser.parse_args()
    # print('Parsed arguments')

    settings = utils.parse_settings(args.settings)
    # print('Parsed settings')
    trueoutputdir = os.path.join(args.outputdir, settings['group'])
    ensure_dir_exists(trueoutputdir)
    # print('Ensured true output directory exists')
    filename = socket.gethostname()+'.'+str(os.getpid())
    runningdir = os.path.join(args.outputdir, 'running')
    ensure_dir_exists(runningdir)
    runningfile = os.path.join(runningdir, filename)
    lda_helper = get_lda_helper(settings['lda_helper'])
    try:
        with open(runningfile, 'w') as outputfh:
            outputfh.write('running')
        # print('Created running mark')
        outprefix = os.path.join(trueoutputdir, args.label)

        start = time.time()
        input_pickle = os.path.join(args.outputdir,
                                    utils.get_pickle_name(args.settings))
        with open(input_pickle, 'rb') as ifh:
            dataset = pickle.load(ifh)
        # print('Got pickle')
        if args.seed == -1:
            rng = random.Random(int(settings['seed']))
        else:
            rng = random.Random(args.seed)
        # print('Set random seed: ', args.seed)
        model = classtm.models.build(rng, settings)
        # print('Built model')
        test_doc_ids, train_doc_ids = partition_data_ids(dataset.num_docs,
                                                         rng,
                                                         settings)
        test_labels = []
        test_words = []
        for tid in test_doc_ids:
            test_labels.append(dataset.labels[dataset.titles[tid]])
            test_words.append(dataset.doc_tokens(tid))
        known_labels = []
        for tid in train_doc_ids:
            known_labels.append(dataset.labels[dataset.titles[tid]])
        # print('Set up initial sets')

        end = time.time()
        init_time = datetime.timedelta(seconds=end-start)

        start = time.time()
        anchors_file = settings.get('anchors_file')
        model.train(dataset, train_doc_ids, known_labels, outprefix,
                    lda_helper, anchors_file)
        end = time.time()
        train_time = datetime.timedelta(seconds=end-start)
        # print('Trained model')

        start = time.time()
        confusion_matrix = evaluate.confusion_matrix(model,
                                                     test_words,
                                                     test_labels,
                                                     dataset.classorder)
        end = time.time()
        eval_time = datetime.timedelta(seconds=end-start)
        model.cleanup()

        with open(outprefix+'.results', 'wb') as ofh:
            pickle.dump({'init_time': init_time,
                         'confusion_matrix': confusion_matrix,
                         'train_time': train_time,
                         'eval_time': eval_time,
                         'model': model},
                        ofh)
    finally:
        os.remove(runningfile)


if __name__ == '__main__':
    _run()

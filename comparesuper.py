"""Code to run experiments for incrementally labeled datasets"""
import argparse
import datetime
import os
import pickle
import random
import socket
import time

from activetm import utils
import classtm.labeled
import classtm.models
from classtm import evaluate

import submain


def parse_args():
    """Parses arguments"""
    parser = argparse.ArgumentParser(
        description='For incrementally labeled experiments')
    parser.add_argument('settings', help=\
            '''the path to a file containing settings, as described in \
            README.md in the root ClassTM directory''')
    parser.add_argument('outputdir', help='directory for output')
    parser.add_argument('label', help='identifying label')
    parser.add_argument('seed', default=-1, type=int, nargs='?')
    return parser.parse_args()


def _run():
    args = parse_args()
    settings = utils.parse_settings(args.settings)
    trueoutputdir = os.path.join(args.outputdir, settings['group'])
    submain.ensure_dir_exists(trueoutputdir)
    # for killing currently running jobs
    filename = socket.gethostname()+'.'+str(os.getpid())
    runningdir = os.path.join(args.outputdir, 'running')
    submain.ensure_dir_exists(runningdir)
    runningfile = os.path.join(runningdir, filename)
    lda_helper = submain.get_lda_helper(settings['lda_helper'])
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
        test_doc_ids, unlabeled_doc_ids = submain.partition_data_ids(
            dataset.num_docs,
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
        model = classtm.models.build(rng, dataset, settings)
        # print('Built model')

        end = time.time()
        init_time = datetime.timedelta(seconds=end-start)

        startlabeled = int(settings['startlabeled'])
        endlabeled = int(settings['endlabeled'])
        increment = int(settings['increment'])
        labeled_count = startlabeled \
            if startlabeled < len(unlabeled_doc_ids) else len(unlabeled_doc_ids)
        train_labels = []
        for docid in unlabeled_doc_ids[:labeled_count]:
            train_labels.append(dataset.labels[dataset.titles[docid]])
        results = []
        while len(incrementaldataset.labels) <= endlabeled:
            start = time.time()
            anchors_file = settings.get('anchors_file')
            model.train(
                dataset,
                unlabeled_doc_ids[:labeled_count],
                train_labels,
                outprefix,
                lda_helper,
                anchors_file)
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
            results.append({'init_time': init_time,
                            'confusion_matrix': confusion_matrix,
                            'labeled_count': labeled_count,
                            'train_time': train_time,
                            'eval_time': eval_time,
                            'model': model})
            if labeled_count >= len(unlabeled_doc_ids):
                break
            prev_count = labeled_count
            if len(unlabeled_doc_ids) >= labeled_count + increment:
                labeled_count += increment
            else:
                labeled_count = len(unlabeled_doc_ids)
            for docid in unlabeled_doc_ids[prev_count:labeled_count]:
                train_labels.append(dataset.labels[dataset.titles[docid]])
        model.cleanup()

        with open(outprefix+'.results', 'wb') as ofh:
            pickle.dump(results, ofh)
    finally:
        os.remove(runningfile)

if __name__ == '__main__':
    _run()

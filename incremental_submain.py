"""Code to run experiments for incrementally labeled datasets"""
import argparse
import datetime
import os
import pickle
import random
import socket
from threading import Thread
import time

from activetm import utils
import classtm.labeled
import classtm.models
from classtm import evaluate

import submain


def kill_at_sixty(outprefix):
    """Kills the program at 60 minutes"""
    # Should sleep 60 minutes, 60*60 is 3600 seconds
    seconds_to_sleep = 3600
    time.sleep(seconds_to_sleep)
    with open(outprefix+'.results', 'wb') as ofh:
        pickle.dump('Hit the 60-minute mark, ran out of time!', ofh)
    print('\n\nHit the 60-minute mark, ran out of time!\n\n')
    os._exit(os.EX_OK)


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
        test_doc_ids, train_doc_ids = submain.partition_data_ids(dataset.num_docs,
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
        model, incrementaldataset = classtm.models.initialize(rng,
                                                              dataset,
                                                              settings)
        # print('Built model')

        end = time.time()
        init_time = datetime.timedelta(seconds=end-start)

        startlabeled = int(settings['startlabeled'])
        endlabeled = int(settings['endlabeled'])
        increment = int(settings['increment'])
        start = time.time()
        initial_train = [dataset.titles[trainid]
                         for trainid in train_doc_ids[:startlabeled]]
        incrementaldataset.initial_label(
            initial_train,
            [dataset.labels[title] for title in initial_train])
        label_time = datetime.timedelta(seconds=time.time()-start)
        results = []
        labeled_count = startlabeled
        # this thread will kill the program after 60 minutes
#        thread = Thread(target=kill_at_sixty,args=(outprefix),daemon=True)
#        thread.start()
        while len(incrementaldataset.labels) <= endlabeled:
            anchors_file = settings.get('anchors_file')
            anchorwords_time, applytrain_time, train_time = model.train(incrementaldataset,
                                                                        outprefix,
                                                                        lda_helper,
                                                                        anchors_file)
            confusion_matrix, devtest_time = evaluate.confusion_matrix(model,
                                                                       test_words,
                                                                       test_labels,
                                                                       dataset.classorder)
            results.append({'init_time': init_time,
                            'confusion_matrix': confusion_matrix,
                            'labeled_count': labeled_count,
                            'anchorwords_time': anchorwords_time,
                            'applytrain_time': applytrain_time,
                            'train_time': train_time,
                            'devtest_time': devtest_time,
                            'label_time': label_time,
                            'model': model})
            if len(incrementaldataset.labels) >= len(train_doc_ids) or \
                    len(incrementaldataset.labels) >= endlabeled:
                break
            start = time.time()
            for i, trainid in enumerate(train_doc_ids[len(incrementaldataset.labels):]):
                if i < increment:
                    labeled_count += 1
                    title = dataset.titles[trainid]
                    incrementaldataset.label_document(title,
                                                      dataset.labels[title])
            label_time = datetime.timedelta(seconds=time.time()-start)
        model.cleanup()

        with open(outprefix+'.results', 'wb') as ofh:
            pickle.dump(results, ofh)
    finally:
        os.remove(runningfile)

if __name__ == '__main__':
    _run()

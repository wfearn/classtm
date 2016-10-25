#! /usr/bin/env python3

"""Pickles the Newsgroups dataset for use with classtm"""

import argparse
import datetime
import os
import pickle
import time

import ankura.label as label
import ankura.pipeline

from classtm.labeled import ClassifiedDataset, get_labels
from activetm import utils


def get_dataset(settings):
    """Get dataset"""
    if settings['corpus'].find('*') >= 0:
        dataset = ankura.pipeline.read_glob(settings['corpus'], labeler=label.text)
    else:
        dataset = ankura.pipeline.read_file(settings['corpus'], labeler=label.text)
    dataset = ankura.pipeline.filter_stopwords(dataset, settings['englstop'])
    dataset = ankura.pipeline.filter_stopwords(dataset, settings['newsstop'])
    dataset = ankura.pipeline.combine_words(dataset, settings['namestop'])
    dataset = ankura.pipeline.filter_rarewords(dataset, int(settings['rare']))
    dataset = ankura.pipeline.filter_commonwords(dataset, int(settings['common']))
    if settings['pregenerate'] == 'YES':
        dataset = ankura.pipeline.pregenerate_doc_tokens(dataset)
    return dataset


def _run():
    parser = argparse.ArgumentParser(description='Pickler of ClassTM datasets')
    parser.add_argument('settings', help='path to a file containing settings')
    parser.add_argument('outputdir', help='directory for output')
    args = parser.parse_args()

    start = time.time()
    settings = utils.parse_settings(args.settings)
    pickle_name = utils.get_pickle_name(args.settings)
    if not os.path.exists(os.path.join(args.outputdir, pickle_name)):
        pre_dataset = get_dataset(settings)
        labels, classorder = get_labels(settings['labels'])
        dataset = ClassifiedDataset(pre_dataset,
                                    labels,
                                    classorder)
        with open(os.path.join(args.outputdir, pickle_name), 'wb') as ofh:
            pickle.dump(dataset, ofh)
    end = time.time()
    import_time = datetime.timedelta(seconds=end-start)
    with open(os.path.join(args.outputdir, pickle_name+'_import.time'), 'w') as ofh:
        ofh.write('# import time: {:s}\n'.format(str(import_time)))


if __name__ == '__main__':
    _run()

from __future__ import print_function

import itertools
import gzip
import json
import resource
import logging
import gc

import ankura


def mem(place):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    collected = gc.collect()
    logging.debug('Place: %s Mem: %i Collected: %i', place, rss, collected)


def read_gzip_json(filename):
    mem('start read_gzip_json')
    with gzip.GzipFile(filename) as docfile:
        for i, line in enumerate(docfile):
            if i % 41 != 0:
                continue
            doc = json.loads(line.decode('utf-8'))
            try:
                yield {'name': doc['reviewerID'] + str(doc['unixReviewTime']),
                       'text': doc['summary'] + ' ' + doc['reviewText'],
                       'label': doc['overall'],
                      }
            except:
                pass
    mem('end read_gzip_json')


def amazon_segmenter(doc_data):
    mem('start amazon_segmenter')
    for doc in doc_data:
        yield doc['name'], doc['text']
    mem('end amazon_segmenter')


def make_amazon_labeler(doc_data):
    mem('start make_amazon_labeler')
    labels = {}
    for doc in doc_data:
        labels[doc['name']] = doc['label']
    def label_fn(name, _):
        return {'label': labels[name]}
    mem('end make_amazon_labeler')
    return label_fn


@ankura.util.memoize
@ankura.util.pickle_cache('amazon_large.pickle')
def get_amazon():
    """Retrieves the gzipped amazon data"""
    data_file = '/aml/home/jlund3/asdf.gz'
    engl_stop = '/local/jlund3/data/stopwords/english.txt'

    tokenizer = ankura.tokenize.simple
    doc_data = amazon_segmenter(read_gzip_json(data_file))
    labeler = make_amazon_labeler(read_gzip_json(data_file))

    mem('pre build_dataset')
    dataset = ankura.pipeline._build_dataset(doc_data, tokenizer, labeler)
    mem('pre filter_stopwords')
    dataset = ankura.filter_stopwords(dataset, engl_stop)
    mem('pre filter_rarewords')
    dataset = ankura.filter_rarewords(dataset, 100)
    mem('pre filter_commonwords')
    dataset = ankura.filter_commonwords(dataset, int(dataset.num_docs * .075))

    return dataset


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    get_amazon()

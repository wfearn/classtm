"""ClassifiedDataset for labeled datasets (classification)"""
import os

import numpy as np
import scipy.sparse

import ankura.pipeline


def get_labels(filename):
    """Reads label information

    Returns examples paired with label and list of different labels
    """
    labels = {}
    with open(filename) as ifh:
        for line in ifh:
            line = line.strip()
            if line:
                example, label = line.split()
                labels[example] = label
    return labels


def get_classorder(labels):
    """Builds classorder from labels (lexicographic order)

        * labels :: {str: str}
            dictionary of document title associated with document label
    """
    tmp = {}
    for _, label in labels.items():
        if label not in tmp:
            tmp[label] = True
    sorted_labels = sorted(tmp.keys())
    classorder = {}
    for i, label in enumerate(sorted_labels):
        classorder[label] = i
    return classorder


def get_newsgroups_labels(dataset):
    """Gets coarse class labels for newsgroups from a dataset object"""
    complabel = ['comp.graphics', 'comp.os.ms-windows.misc',
                 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
                 'comp.windows.x']
    reclabel = ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball',
                'rec.sport.hockey']
    scilabel = ['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space']
    forsale = ['misc.forsale']
    polilabel = ['talk.politics.guns', 'talk.politics.mideast',
                 'talk.politics.misc']
    rellabel = ['talk.religion.misc', 'alt.atheism', 'soc.religion.christian']
    labels = {}
    classorder = {}
    classindex = 0
    for title, metadatum in zip(dataset.titles, dataset.metadata):
        pre_label = os.path.split(metadatum['dirname'])[1]
        label = None
        if pre_label in complabel:
            label = 'computer'
        elif pre_label in reclabel:
            label = 'recreation'
        elif pre_label in scilabel:
            label = 'science'
        elif pre_label in forsale:
            label = 'forsale'
        elif pre_label in polilabel:
            label = 'politics'
        elif pre_label in rellabel:
            label = 'religion'
        if label not in classorder:
            classorder[label] = classindex
            classindex += 1
        labels[title] = label
    return labels, classorder


def orderclasses(classorder):
    """Build list of classes, indexed according to classorder"""
    orderedclasses = [''] * len(classorder)
    for cla in classorder:
        orderedclasses[classorder[cla]] = cla
    return orderedclasses


class AbstractClassifiedDataset(ankura.pipeline.Dataset):
    """For use with classtm models"""

    def __init__(self, dataset, labels, classorder):
        super(AbstractClassifiedDataset, self).__init__(dataset.docwords,
                                                        dataset.vocab,
                                                        dataset.titles,
                                                        metadata=dataset.metadata)
        self.labels = labels
        self.classorder = classorder
        self.orderedclasses = orderclasses(self.classorder)


#pylint:disable-msg=too-few-public-methods
class ClassifiedDataset(AbstractClassifiedDataset):
    """Classified, as in data is labeled with classes"""

    def __init__(self, dataset, labels, classorder):
        super(ClassifiedDataset, self).__init__(dataset, labels, classorder)
        # add pseudo labels if necessary
        if not isinstance(dataset, ClassifiedDataset):
            self.origvocabsize = len(self._vocab)
            self._vocab = np.append(self._vocab, self.orderedclasses)
            tmp = scipy.sparse.lil_matrix((len(self._vocab), len(self.titles)),
                                          dtype=self._docwords.dtype)
            tmp[:self.origvocabsize, :] = self._docwords
            for docnum, title in enumerate(self.titles):
                label = self.labels[title]
                tmp[self.origvocabsize+self.classorder[label], docnum] = 1
            self._docwords = tmp.tocsc()
        # when compute_cooccurrences gets called, we should get the Q we want

    def doc_tokens(self, doc_id, rng=np.random):
        if doc_id in self._tokens:
            return self._tokens[doc_id]

        token_ids, _, counts = scipy.sparse.find(self._docwords[:, doc_id])
        tokens = []
        for token_id, count in zip(token_ids, counts):
            if token_id < self.origvocabsize:
                tokens.extend([token_id] * count)
        #pylint:disable-msg=no-member
        rng.shuffle(tokens)

        self._tokens[doc_id] = tokens
        return tokens


def get_titles_order(titles):
    """Builds dictionary mapping title to index"""
    result = {}
    for i, title in enumerate(titles):
        result[title] = i
    return result


def doc_scaled(value):
    """Returns function that returns value scaled by document length"""
    def _inner(doclength, corpussize):
        """Returns value scaled by doclength"""
        del corpussize
        return doclength * value
    return _inner


def corpus_scaled(value):
    """Returns function that returns value scaled by corpus size"""
    def _inner(doclength, corpussize):
        """Returns value scaled by corpus size"""
        del doclength
        return corpussize * value
    return _inner


def return_value(value):
    """Returns function that returns value"""
    def _inner(doclength, corpussize):
        """Returns value"""
        del doclength, corpussize
        return value
    return _inner


def get_label_weight_function(label_weight):
    """Get function that returns label weight function

        * label_weight :: str
            label_weight can take one of three forms:
                * doc:<float>
                * corpus:<float>
                * <float>
    """
    if ':' in label_weight:
        args = label_weight.split(':')
        if args[0] == 'doc':
            return doc_scaled(float(args[1]))
        elif args[0] == 'corpus':
            return corpus_scaled(float(args[1]))
        else:
            raise Exception('Unknown label_weight function: ' + args[0])
    return return_value(float(label_weight))


#pylint:disable-msg=too-many-instance-attributes
class IncrementalClassifiedDataset(AbstractClassifiedDataset):
    """ClassifiedDataset for incremental case"""

    def __init__(self, dataset, settings):
        super(IncrementalClassifiedDataset, self).__init__(dataset, {}, {})
        self.origvocabsize = len(self._vocab)
        self.smoothing = float(settings['smoothing'])
        self.label_weight = get_label_weight_function(settings['label_weight'])
        self.titlesorder = get_titles_order(self.titles)

    def doc_tokens(self, doc_id, rng=np.random):
        if doc_id in self._tokens:
            return self._tokens[doc_id]

        token_ids, _, counts = scipy.sparse.find(self._docwords[:, doc_id])
        tokens = []
        for token_id, count in zip(token_ids, counts):
            if token_id < self.origvocabsize:
                tokens.extend([token_id] * count)
        #pylint:disable-msg=no-member
        rng.shuffle(tokens)

        self._tokens[doc_id] = tokens
        return tokens

    def _label_helper(self, docwords, title, label):
        docnum = self.titlesorder[title]
        # erase smoothing on labeled documents
        docwords[self.origvocabsize:, docnum] = 0
        # apply label
        docwords[self.origvocabsize+self.classorder[label], docnum] = \
            self.label_weight(docwords[:self.origvocabsize, docnum].sum(),
                              docwords.shape[1])

    def label_document(self, title, label):
        """Label a document in this corpus

            * title :: str
                title of document
            * label :: str
                label of document
        Assumes that title is in corpus; rebuilds docwords matrix (it not done
        after every labeling, indexing problems crop up on predict)
        """
        self.labels[title] = label
        if label not in self.classorder:
            self.classorder[label] = len(self.classorder)
            self.orderedclasses = orderclasses(self.classorder)
            self._vocab = np.append(self._vocab, label)
            # resize docwords matrix only when number of classes increases
            tmp = scipy.sparse.lil_matrix((len(self._vocab), len(self.titles)),
                                          dtype=self._docwords.dtype)
            tmp[:self.origvocabsize, :] = self._docwords[:self.origvocabsize, :]
            tmp[self.origvocabsize:, :] = self.smoothing
            for curtitle, curlabel in self.labels.items():
                self._label_helper(tmp, curtitle, curlabel)
            self._docwords = tmp.tocsc()
        else:
            # number of classes remain the same, so just update the one column
            # that needs updating
            tmp = self._docwords.tolil()
            self._label_helper(self._docwords, title, label)
            self._docwords = tmp
        self._cooccurrences = None


class SupervisedAnchorDataset(AbstractClassifiedDataset):
    """Dataset implementing Nguyen et al. (NAACL 2015)"""

    def __init__(self, dataset, labels, classorder):
        super(SupervisedAnchorDataset, self).__init__(dataset,
                                                      labels,
                                                      classorder)
        # precompute \bar{Q}
        ankura.pipeline.Dataset.compute_cooccurrences(self)
        self._dataset_cooccurrences = self._cooccurrences
        # fool ankura into calling compute_cooccurrences
        self._cooccurrences = None

    def compute_cooccurrences(self):
        orig_height, orig_width = self._dataset_cooccurrences.shape
        classcount = len(self.classorder)
        self._cooccurrences = np.zeros((orig_height, orig_width+classcount))
        self._cooccurrences[:, :-classcount] = self._dataset_cooccurrences
        # assuming that self._docwords is an instance of a scipy sparse matrix
        docwords_csr = self._docwords.tocsr()
        indices = docwords_csr.indices
        indptr = docwords_csr.indptr
        data = docwords_csr.data
        for i in range(orig_height):
            total = 0
            for docnum, datum in zip(indices[indptr[i]:indptr[i+1]],
                                     data[indptr[i]:indptr[i+1]]):
                if datum > 0:
                    # tally up number documents with class label
                    label_string = self.labels.get(self.titles[docnum])
                    if label_string:
                        # count up number of labeled documents with word i
                        total += 1
                        label = self.classorder[label_string]
                        self._cooccurrences[i, orig_width+label] += 1
            # normalize tally
            if total > 0:
                self._cooccurrences[i, -classcount:] /= total


class IncrementalSupervisedAnchorDataset(SupervisedAnchorDataset):
    """Dataset implementing supervised anchor words but for use with
    incrementally labeled data
    """

    def __init__(self, dataset, _):
        super(IncrementalSupervisedAnchorDataset, self).__init__(dataset,
                                                                 {},
                                                                 {})
        self.titlesorder = get_titles_order(self.titles)

    def label_document(self, title, label):
        """Label a document in this corpus

            * title :: str
                title of document
            * label :: str
                label of document
        Assumes that title is in corpus
        """
        self.labels[title] = label
        if label not in self.classorder:
            self.classorder[label] = len(self.classorder)
            self.orderedclasses = orderclasses(self.classorder)
        self._cooccurrences = None


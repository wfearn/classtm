"""ClassifiedDataset for labeled datasets (classification)"""
import ctypes
import os

import numpy as np
import numpy.ctypeslib as npct
import scipy.sparse

import ankura.pipeline


ARRAY_1D_DOUBLE = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
SO_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'simplex',
    'simplexproj.so')
LIBCD = ctypes.CDLL(SO_PATH)
LIBCD.simplexproj.restype = None
LIBCD.simplexproj.argtypes = [
    ARRAY_1D_DOUBLE,
    ARRAY_1D_DOUBLE,
    ctypes.c_int,
    ctypes.c_double]


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
        super(AbstractClassifiedDataset, self).__init__(
            dataset.docwords,
            dataset.vocab,
            dataset.titles,
            metadata=dataset.metadata)
        self.labels = labels
        self.classorder = classorder
        self.orderedclasses = orderclasses(self.classorder)


class AbstractParameterizedClassifiedDataset(AbstractClassifiedDataset):
    """When you want parameters on how Q gets constructed"""

    def __init__(self, dataset, labels, classorder, smoothing, label_weight):
        super(AbstractParameterizedClassifiedDataset, self).__init__(
            dataset,
            labels,
            classorder)
        self.smoothing = smoothing
        self.label_weight = get_label_weight_function(label_weight)


# pylint:disable-msg=too-few-public-methods
class ClassifiedDataset(AbstractParameterizedClassifiedDataset):
    """Classified, as in data is labeled with classes"""

    def __init__(self, dataset, labels, classorder, smoothing, label_weight):
        super(ClassifiedDataset, self).__init__(dataset,
                                                labels,
                                                classorder,
                                                smoothing,
                                                label_weight)
        # add pseudo labels if necessary
        if not isinstance(dataset, ClassifiedDataset):
            self.origvocabsize = len(self._vocab)
            self._vocab = np.append(self._vocab, self.orderedclasses)
            tmp = scipy.sparse.lil_matrix((len(self._vocab), len(self.titles)),
                                          dtype=self._docwords.dtype)
            tmp[:self.origvocabsize, :] = self._docwords
            tmp[self.origvocabsize:, :] = self.smoothing
            for docnum, title in enumerate(self.titles):
                label = self.labels[title]
                tmp[self.origvocabsize+self.classorder[label], docnum] = \
                    self.label_weight(
                        self._docwords[:self.origvocabsize, docnum].sum(),
                        self._docwords.shape[1])
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
        # pylint:disable-msg=no-member
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


# pylint:disable-msg=too-many-instance-attributes
class IncrementalClassifiedDataset(AbstractParameterizedClassifiedDataset):
    """ClassifiedDataset for incremental case"""

    def __init__(self, dataset, settings):
        smoothing = float(settings['smoothing'])
        label_weight = settings['label_weight']
        super(IncrementalClassifiedDataset, self).__init__(dataset,
                                                           {},
                                                           {},
                                                           smoothing,
                                                           label_weight)
        self.origvocabsize = len(self._vocab)
        self.titlesorder = get_titles_order(self.titles)

    def doc_tokens(self, doc_id, rng=np.random):
        if doc_id in self._tokens:
            return self._tokens[doc_id]

        token_ids, _, counts = scipy.sparse.find(self._docwords[:, doc_id])
        tokens = []
        for token_id, count in zip(token_ids, counts):
            if token_id < self.origvocabsize:
                tokens.extend([token_id] * count)
        # pylint:disable-msg=no-member
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

    def initial_label(self, titles, labels):
        """Account for initially labeled documents

            * titles :: [str]
                titles of documents to be labeled
            * labels :: [str]
                labels of documents
        Assumes that titles are in corpus and that there are no labeled
        documents currently in the corpus.  Also assumes that self._docwords is
        a scipy.sparse.csc_matrix.
        """
        newlabels = []
        for title, label in zip(titles, labels):
            self.labels[title] = label
            if label not in self.classorder:
                newlabels.append(label)
                self.classorder[label] = len(self.classorder)
        self._vocab = np.append(self._vocab, newlabels)
        self.orderedclasses = orderclasses(self.classorder)
        data = []
        indices = []
        indptr = [0]
        data_tmp = [self.smoothing] * len(self.classorder)
        indices_tmp = [i for i in range(self.origvocabsize, len(self._vocab))]
        for start, stop, title in zip(self._docwords.indptr[:-1],
                                      self._docwords.indptr[1:],
                                      self.titles):
            # copy data from original docwords matrix
            data.extend(self._docwords.data[start:stop])
            indices.extend(self._docwords.indices[start:stop])
            if title in self.labels:
                # label document if labeled
                label = self.labels[title]
                data.append(self.label_weight(
                    sum(self._docwords.data[start:stop]),
                    self._docwords.shape[1]))
                indices.append(self.origvocabsize+self.classorder[label])
            else:
                # otherwise, apply smoothing
                data.extend(data_tmp)
                indices.extend(indices_tmp)
            indptr.append(len(data))
        self._docwords = scipy.sparse.csc_matrix((data, indices, indptr),
                                                 shape=(len(self._vocab),
                                                        len(self.titles)))
        self.compute_cooccurrences()

    def label_document(self, title, label):
        """Label a document in this corpus

            * title :: str
                title of document
            * label :: str
                label of document
        Assumes that title is in corpus; rebuilds docwords matrix (if not done
        after every labeling, indexing problems crop up on predict)
        """
        self.labels[title] = label
        if label not in self.classorder:
            self.classorder[label] = len(self.classorder)
            self.orderedclasses = orderclasses(self.classorder)
            self._vocab = np.append(self._vocab, label)
            # resize docwords matrix only when number of classes increases
            tmp = scipy.sparse.lil_matrix((len(self._vocab), len(self.titles)),
                                          dtype=np.float)
            tmp[:self.origvocabsize, :] = \
                self._docwords[:self.origvocabsize, :]
            tmp[self.origvocabsize:, :] = self.smoothing
            for curtitle, curlabel in self.labels.items():
                self._label_helper(tmp, curtitle, curlabel)
            self._docwords = tmp.tocsc()
        else:
            # number of classes remain the same, so just update the one column
            # that needs updating
            tmp = self._docwords.tolil()
            self._label_helper(tmp, title, label)
            self._docwords = tmp.tocsc()
        self._cooccurrences = None


class QuickIncrementalClassifiedDataset(IncrementalClassifiedDataset):
    """ClassifiedDataset for incremental labeling using quick Q building"""

    def __init__(self, dataset, settings):
        super(QuickIncrementalClassifiedDataset, self).__init__(dataset,
                                                                settings)
        self.newlabels = {}
        self.prevq = None

    # pylint:disable-msg=invalid-name
    def _build_miniq(self, docnums):
        """Builds Q with just the documents in docnums

            * docnums :: np.array
                column indices into self._docwords for the documents that have
                been labeled for this update
        """
        data = []
        indices = []
        indptr = [0]
        H_hat = np.zeros(self.vocab_size)
        for docnum in docnums:
            col_start = self._docwords.indptr[docnum]
            col_end = self._docwords.indptr[docnum+1]
            row_indices = self._docwords.indices[col_start:col_end]
            count = np.sum(self._docwords.data[col_start:col_end])
            norm = count * (count - 1)
            if norm != 0:
                sqrtnorm = np.sqrt(norm)
                H_hat[row_indices] += \
                    self._docwords.data[col_start:col_end] / norm
                data.extend(self._docwords.data[col_start:col_end] / sqrtnorm)
                indices.extend(row_indices)
                indptr.append(len(data))
        H_tilde = scipy.sparse.csc_matrix(
            (data, indices, indptr),
            shape=(self.vocab_size, len(indptr)-1),
            dtype=np.float)
        return H_tilde * H_tilde.transpose() - np.diag(H_hat)

    def compute_cooccurrences(self, epsilon=1e-15):
        """Updates Q"""
        if self.prevq is None:
            ankura.pipeline.Dataset.compute_cooccurrences(self, epsilon)
            self.prevq = self._cooccurrences
        else:
            # reload previous Q
            self._cooccurrences = self.prevq
        if self.newlabels:
            # compute what needs to be taken out of Q
            docnums = []
            for title in self.newlabels:
                docnums.append(self.titlesorder[title])
            docnums = np.array(docnums)
            miniq = self._build_miniq(docnums)
            # take it out of Q
            self._cooccurrences -= np.array(miniq / self._docwords.shape[1])
            # compute what needs to be put into Q
            tmp = self._docwords.tolil()
            for title, label in self.newlabels.items():
                self._label_helper(tmp, title, label)
            self._docwords = tmp.tocsc()
            miniq = self._build_miniq(docnums)
            # put it into Q
            self._cooccurrences += np.array(miniq / self._docwords.shape[1])
            # reset new labels
            self.newlabels = {}
        if np.any(self._cooccurrences < 0):
            print('Negative in Q')
            print(np.transpose(np.nonzero(self._cooccurrences < 0)))
            print(self._cooccurrences[self._cooccurrences < 0])
            print('Original vocab size:', self.origvocabsize, flush=True)
        self._cooccurrences[
            (-epsilon < self._cooccurrences) & (self._cooccurrences < 0)] = 0

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
            # add labels for previously labeled documents
            self.compute_cooccurrences()
            # need to expand Q, since there is now a new label; let
            # compute_cooccurrences get the right values for smoothing terms
            self.prevq = None
            # add new label
            self.classorder[label] = len(self.classorder)
            self.orderedclasses = orderclasses(self.classorder)
            self._vocab = np.append(self._vocab, label)
            # resize docwords matrix only when number of classes increases
            tmp = scipy.sparse.lil_matrix((len(self._vocab), len(self.titles)),
                                          dtype=np.float)
            tmp[:self.origvocabsize, :] = \
                self._docwords[:self.origvocabsize, :]
            tmp[self.origvocabsize:, :] = self.smoothing
            for curtitle, curlabel in self.labels.items():
                self._label_helper(tmp, curtitle, curlabel)
            self._docwords = tmp.tocsc()
        else:
            self.newlabels[title] = label
        self._cooccurrences = None


class ProjectedDataset(QuickIncrementalClassifiedDataset):
    """Project Q matrix to simplex"""

    def __init__(self, dataset, settings):
        super(ProjectedDataset, self).__init__(dataset,
                                               settings)

    @property
    def Q(self):
        result = super(ProjectedDataset, self).Q.copy()
        if np.any(result < 0):
            result = self._project(result)
        return result

    def compute_cooccurrences(self, epsilon=1e-15):
        """Updates Q"""
        if self.prevq is None:
            ankura.pipeline.Dataset.compute_cooccurrences(self, epsilon)
            self.prevq = self._cooccurrences
        else:
            # reload previous Q
            self._cooccurrences = self.prevq
        if self.newlabels:
            # compute what needs to be taken out of Q
            docnums = []
            for title in self.newlabels:
                docnums.append(self.titlesorder[title])
            docnums = np.array(docnums)
            miniq = self._build_miniq(docnums)
            # take it out of Q
            self._cooccurrences -= np.array(miniq / self._docwords.shape[1])
            # compute what needs to be put into Q
            tmp = self._docwords.tolil()
            for title, label in self.newlabels.items():
                self._label_helper(tmp, title, label)
            self._docwords = tmp.tocsc()
            miniq = self._build_miniq(docnums)
            # put it into Q
            self._cooccurrences += np.array(miniq / self._docwords.shape[1])
            # reset new labels
            self.newlabels = {}

    def _project(self, vector):
        """Projects vector onto simplex

        Uses algorithm proposed by Condat in "Fast Projection onto the Simplex
        and the l_1 Ball" (Mathematical Programming, July 2016, vol. 158, iss.
        1)
        """
        flattened = vector.ravel()
        LIBCD.simplexproj(flattened, flattened, flattened.size, 1.0)
        return flattened.reshape(vector.shape)


class ZeroNegativesDataset(QuickIncrementalClassifiedDataset):
    """Zero out negative values in Q"""

    def __init__(self, dataset, settings):
        super(ZeroNegativesDataset, self).__init__(dataset,
                                                   settings)

    @property
    def Q(self):
        result = super(ZeroNegativesDataset, self).Q.copy()
        result[result < 0] = 0
        return result


class ZeroEpsilonDataset(QuickIncrementalClassifiedDataset):
    """Quick check to see if zeroing idea works"""

    def __init__(self, dataset, settings):
        super(ZeroEpsilonDataset, self).__init__(dataset,
                                                 settings)

    # pylint:disable-msg=invalid-name
    def _build_miniq(self, docnums):
        """Builds Q with just the documents in docnums

            * docnums :: np.array
                column indices into self._docwords for the documents that have
                been labeled for this update
        """
        data = []
        indices = []
        indptr = [0]
        H_hat = np.zeros(self.vocab_size)
        for docnum in docnums:
            col_start = self._docwords.indptr[docnum]
            col_end = self._docwords.indptr[docnum+1]
            row_indices = self._docwords.indices[col_start:col_end]
            count = np.sum(self._docwords.data[col_start:col_end])
            norm = count * (count - 1)
            if norm != 0:
                sqrtnorm = np.sqrt(norm)
                labels_start = col_end - len(self.classorder)
                if np.all(
                        self._docwords.data[labels_start:col_end] ==
                        self.smoothing):
                    H_hat[row_indices[:-len(self.classorder)]] += \
                        self._docwords.data[col_start:labels_start] / norm
                    H_hat[row_indices[-len(self.classorder):]] += \
                        np.square(self._docwords.data[labels_start:col_end]) \
                        / norm
                else:
                    H_hat[row_indices] += \
                        self._docwords.data[col_start: col_end] / norm
                data.extend(self._docwords.data[col_start:col_end] / sqrtnorm)
                indices.extend(row_indices)
                indptr.append(len(data))
        H_tilde = scipy.sparse.csc_matrix(
            (data, indices, indptr),
            shape=(self.vocab_size, len(indptr)-1),
            dtype=np.float)
        return H_tilde * H_tilde.transpose() - np.diag(H_hat)

    def init_compute_cooccurrences(self):
        """Initialize Q"""
        vocab_size, num_docs = self.M.shape
        H_tilde = scipy.sparse.csc_matrix(self.M.copy(), dtype=float)
        H_hat = np.zeros(vocab_size)

        # Construct H_tilde and H_hat
        for j in range(H_tilde.indptr.size - 1):
            # get indices of column j
            col_start = H_tilde.indptr[j]
            col_end = H_tilde.indptr[j + 1]
            row_indices = H_tilde.indices[col_start: col_end]

            # get count of tokens in column (document) and compute norm
            count = np.sum(H_tilde.data[col_start: col_end])
            norm = count * (count - 1)

            # update H_hat and H_tilde (see supplementary)
            if norm != 0:
                labels_start = col_end - len(self.classorder)
                if len(self.classorder) and np.all(
                        H_tilde.data[labels_start:col_end] == self.smoothing):
                    H_hat[row_indices[:-len(self.classorder)]] += \
                        H_tilde.data[col_start:labels_start] / norm
                    H_hat[row_indices[-len(self.classorder):]] += \
                        np.square(H_tilde.data[labels_start:col_end]) / norm
                else:
                    H_hat[row_indices] += \
                        H_tilde.data[col_start: col_end] / norm
                H_tilde.data[col_start: col_end] /= np.sqrt(norm)

        # construct and store normalized Q
        Q = H_tilde * H_tilde.transpose() - np.diag(H_hat)
        self._cooccurrences = np.array(Q / num_docs)

    def compute_cooccurrences(self, epsilon=1e-15):
        """Updates Q"""
        if self.prevq is None:
            self.init_compute_cooccurrences()
            self.prevq = self._cooccurrences
        else:
            # reload previous Q
            self._cooccurrences = self.prevq
        if self.newlabels:
            # compute what needs to be taken out of Q
            docnums = []
            for title in self.newlabels:
                docnums.append(self.titlesorder[title])
            docnums = np.array(docnums)
            miniq = self._build_miniq(docnums)
            # take it out of Q
            self._cooccurrences -= np.array(miniq / self._docwords.shape[1])
            # compute what needs to be put into Q
            tmp = self._docwords.tolil()
            for title, label in self.newlabels.items():
                self._label_helper(tmp, title, label)
            self._docwords = tmp.tocsc()
            miniq = self._build_miniq(docnums)
            # put it into Q
            self._cooccurrences += np.array(miniq / self._docwords.shape[1])
            # reset new labels
            self.newlabels = {}
        if np.any(self._cooccurrences < 0):
            print('Negative in Q')
            print(np.transpose(np.nonzero(self._cooccurrences < 0)))
            print(self._cooccurrences[self._cooccurrences < 0])
            print('Original vocab size:', self.origvocabsize, flush=True)
        self._cooccurrences[
            (-epsilon < self._cooccurrences) & (self._cooccurrences < 0)] = 0


class SupervisedAnchorDataset(AbstractClassifiedDataset):
    """Dataset implementing Nguyen et al. (NAACL 2015)"""

    def __init__(self, dataset, labels, classorder):
        super(SupervisedAnchorDataset, self).__init__(dataset,
                                                      labels,
                                                      classorder)
        # precompute \bar{Q}
        ankura.pipeline.Dataset.compute_cooccurrences(self)
        # np doesn't broadcast across rows, so we make the rows into columns
        # to perform the proper normalization before turning the columns back
        # into rows
        self._dataset_cooccurrences = \
            (self._cooccurrences.T / self._cooccurrences.T.sum(axis=0)).T
        # fool ankura into calling compute_cooccurrences
        self._cooccurrences = None

    def compute_cooccurrences(self, epsilon=1e-15):
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
        self.extra_counts = None
        self.newlabels = {}

    def compute_cooccurrences(self, epsilon=1e-15):
        orig_height, orig_width = self._dataset_cooccurrences.shape
        classcount = len(self.classorder)
        self._cooccurrences = np.zeros((orig_height, orig_width+classcount))
        self._cooccurrences[:, :-classcount] = self._dataset_cooccurrences
        if not self.extra_counts:
            self.extra_counts = np.zeros((orig_height, classcount))
            for title, label in self.newlabels.items():
                self.labels[title] = label
            # assuming that self._docwords is an instance of a scipy sparse
            # matrix
            docwords_csr = self._docwords.tocsr()
            indices = docwords_csr.indices
            indptr = docwords_csr.indptr
            data = docwords_csr.data
            for i in range(orig_height):
                for docnum, datum in zip(indices[indptr[i]:indptr[i+1]],
                                         data[indptr[i]:indptr[i+1]]):
                    if datum > 0:
                        # tally up number documents with class label
                        label_string = self.labels.get(self.titles[docnum])
                        if label_string:
                            # count up number of labeled documents with word i
                            label = self.classorder[label_string]
                            self.extra_counts[i, label] += 1
        else:
            docwords_csc = self._docwords.tocsc()
            indices = docwords_csc.indices
            indptr = docwords_csc.indptr
            data = docwords_csc.data
            for title, label_string in self.newlabels.items():
                title_index = self.titlesorder[title]
                label = self.classorder[label_string]
                if title in self.labels:
                    # document has been re-labeled, so remove counts for label
                    prev_label = self.classorder[self.labels[title]]
                    for word, count in zip(
                            indices[indptr[title_index]:indptr[title_index+1]],
                            data[indptr[title_index]:indptr[title_index+1]]):
                        if count > 0:
                            self.extra_counts[word, prev_label] -= 1
                for word, count in zip(
                        indices[indptr[title_index]:indptr[title_index+1]],
                        data[indptr[title_index]:indptr[title_index+1]]):
                    if count > 0:
                        self.extra_counts[word, label] += 1
        # normalize tally
        row_sums = self.extra_counts.sum(axis=1, keepdims=True)
        # prevent divisions by zero
        row_sums[row_sums == 0] = 1
        self._cooccurrences[:, -classcount:] = self.extra_counts / row_sums
        self.newlabels = {}

    def initial_label(self, titles, labels):
        """Account for initially labeled documents

            * titles :: [str]
                titles of documents to be labeled
            * labels :: [str]
                labels of documents
        Assumes that titles are in corpus and that there are no labeled
        documents currently in the corpus.
        """
        for title, label in zip(titles, labels):
            self.labels[title] = label
            if label not in self.classorder:
                self.classorder[label] = len(self.classorder)
        self.orderedclasses = orderclasses(self.classorder)
        self._cooccurrences = None

    def label_document(self, title, label):
        """Label a document in this corpus

            * title :: str
                title of document
            * label :: str
                label of document
        Assumes that title is in corpus
        """
        self.newlabels[title] = label
        if label not in self.classorder:
            self.classorder[label] = len(self.classorder)
            self.orderedclasses = orderclasses(self.classorder)
        self._cooccurrences = None


class IncrementalSupervisedNormalizedAnchorDataset(
        IncrementalSupervisedAnchorDataset):
    """Dataset for incremental supervised anchor words, but each column of Q is
    weighted to have equal weighting (the original anchor words algorithm very
    heavily weights the last columns, which contain label information)
    """

    def __init__(self, dataset, _):
        super(IncrementalSupervisedNormalizedAnchorDataset, self).__init__(
            dataset,
            {})

    def compute_cooccurrences(self, epsilon=1e-15):
        orig_height, orig_width = self._dataset_cooccurrences.shape
        classcount = len(self.classorder)
        super(IncrementalSupervisedNormalizedAnchorDataset,
            self).compute_cooccurrences(epsilon)
        self._cooccurrences[:, :-classcount] *= \
            orig_width / (orig_width+classcount)
        self._cooccurrences[:, -classcount:] *= classcount /\
            (orig_width+classcount)


SUPANCH_CTORS = [
    SupervisedAnchorDataset,
    IncrementalSupervisedAnchorDataset,
    IncrementalSupervisedNormalizedAnchorDataset]

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


def get_doc_co_counts(sparsevec):
    """Returns document cooccurrence counts as dense matrix

        * sparsevec :: vocab x 1 scipy.sparse.csc_matrix
    """
    first_term = sparsevec * sparsevec.T
    # first_term - Diag(sparsevec)
    # since sparsevec is a column vector, we know that all of its data are in
    # the one column
    for row, value in zip(sparsevec.indices, sparsevec.data):
        for i, ft_row in enumerate(
                first_term.indices[
                    first_term.indptr[row]:first_term.indptr[row+1]]):
            if ft_row == row:
                first_term.data[first_term.indptr[row]+i] -= value
                break
            continue
    return first_term.todense()


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
        result = super(ProjectedDataset, self).Q
        result = self._project(result)
        if np.any(result < 0):
            print('Negative in Q')
            print(np.transpose(np.nonzero(self._cooccurrences < 0)))
            print(self._cooccurrences[self._cooccurrences < 0], flush=True)
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
        dim_max = 1.0
        flattened = vector.ravel()
        greaters = [flattened[0]]
        potentials = []
        rho = flattened[0] - dim_max
        for value in flattened[1:]:
            if value > rho:
                rho += (value - rho) / (len(greaters) + 1)
                if rho > (value - dim_max):
                    greaters.append(value)
                else:
                    potentials.extend(greaters)
                    greaters = [value]
                    rho = value - dim_max
        if potentials:
            for value in potentials:
                if value > rho:
                    greaters.append(value)
                    rho += (value - rho) / len(greaters)
        while True:
            prev_length = len(greaters)
            next_greaters = []
            for i, value in enumerate(greaters):
                if value <= rho:
                    next_greaters.append(value)
                    rho += \
                        (rho - value) / (prev_length - i + len(next_greaters))
            greaters = next_greaters
            if prev_length == len(greaters):
                break
        vector -= rho
        vector[vector < 0] = 0
        return vector


class ZeroNegativesDataset(QuickIncrementalClassifiedDataset):
    """Zero out negative values in Q"""

    def __init__(self, dataset, settings):
        super(ZeroNegativesDataset, self).__init__(dataset,
                                                   settings)

    def compute_cooccurrences(self, epsilon=1e-15):
        super(ZeroNegativesDataset, self).compute_cooccurrences(epsilon)
        self._cooccurrences[self._cooccurrences < 0] = 0


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


SUPANCH_CTORS = [
    SupervisedAnchorDataset,
    IncrementalSupervisedAnchorDataset]

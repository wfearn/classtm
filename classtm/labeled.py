"""ClassifiedDataset for labeled datasets (classification)"""
import scipy.sparse
import numpy as np

import ankura.pipeline


def get_labels(filename):
    """Reads label information

    Returns examples paired with label and list of different labels
    """
    labels = {}
    classorder = {}
    classindex = 0
    with open(filename) as ifh:
        for line in ifh:
            line = line.strip()
            if line:
                example, label = line.split()
                if label not in classorder:
                    classorder[label] = classindex
                    classindex += 1
                labels[example] = label
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
                                                        dataset.titles)
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
        rng.shuffle(tokens)

        self._tokens[doc_id] = tokens
        return tokens


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
                    # count up number of documents with word i
                    total += 1
                    # tally up number odcuments with class label
                    label = self.classorder[self.labels[self.titles[docnum]]]
                    self._cooccurrences[i, orig_width+label] += 1
                # normalize tally
                self._cooccurrences[i, :-classcount] /= total


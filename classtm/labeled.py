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


class ClassifiedDataset(ankura.pipeline.Dataset):
    """Classified, as in data is labeled with classes"""

    def __init__(self, dataset, labels, classorder):
        ankura.pipeline.Dataset.__init__(self,
                                         dataset.docwords,
                                         dataset.vocab,
                                         dataset.titles)
        self.labels = labels
        self.classorder = classorder
        self.orderedclasses = [''] * len(self.classorder)
        for cla in self.classorder:
            self.orderedclasses[self.classorder[cla]] = cla
        # add pseudo labels if necessary
        if not isinstance(dataset, ClassifiedDataset):
            origvocabsize = len(self._vocab)
            self._vocab = np.append(self._vocab, self.orderedclasses)
            tmp = scipy.sparse.lil_matrix((len(self._vocab), len(self.titles)),
                                          dtype=self._docwords.dtype)
            tmp[:origvocabsize, :] = self._docwords
            for docnum, title in enumerate(self.titles):
                label = self.labels[title]
                tmp[origvocabsize+self.classorder[label], docnum] = 1
            self._docwords = tmp.tocsc()
        # when compute_cooccurrences gets called, we should get the Q we want


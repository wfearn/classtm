"""ClassifiedDataset for labeled datasets (classification)"""
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
    """Classified, as in data is labeled with classes

    This implements for supervised anchor words per Nguyen et al., NAACL 2015.
    Namely, it adds the p(y=l|w=i) columns.  In extension to Nguyen et al., the
    Q matrix is also augmented with additional rows.  How these rows are
    calculated is determined by the filladdrowsopt passed into the initializer
    """

    def __init__(self, dataset, labels, classorder, filladdrowsopt):
        ankura.pipeline.Dataset.__init__(self,
                                         dataset.docwords,
                                         dataset.vocab,
                                         dataset.titles)
        self.labels = labels
        self.classorder = classorder
        self.filladdrowsfunc = None
        self.setfilladdrowfunc(filladdrowsopt)
        # precompute vanilla Q
        ankura.pipeline.Dataset.compute_cooccurrences(self)
        self._dataset_cooccurrences = self._cooccurrences
        # fool ankura into calling compute_cooccurrences to construct augmented
        # Q matrix
        self._cooccurrences = None

    def compute_cooccurrences(self):
        orig_height, orig_width = self._dataset_cooccurrences.shape
        classcount = len(self.classorder)
        self._cooccurrences = np.zeros((orig_height+classcount,
                                        orig_width+classcount))
        self._cooccurrences[:-classcount, :-classcount] = \
            self._dataset_cooccurrences
        # lower right block might not be identity if class labels are dependent
        # on each other
        self._cooccurrences[orig_height:, orig_width:] = np.eye(classcount)
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
                    # count up the number of documents with word i
                    total += 1
                    # tally up number of documents with class label
                    label = self.classorder[self.labels[self.titles[docnum]]]
                    self._cooccurrences[i, orig_width+label] += 1
            # normalize tally
            self._cooccurrences[i, orig_width:] /= total
        self.filladdrowsfunc(self)

    def setfilladdrowfunc(self, filladdrowsopt):
        """Sets the function used to fill in the additional rows in Q"""
        if filladdrowsopt == 'tranpose':
            self.filladdrowsfunc = self._transpose_columns
        elif filladdrowsopt == 'flip':
            self.filladdrowsfunc = self._flip_conditional
        else:
            raise NotImplementedError('Unknown option '+filladdrowsopt)

    def _transpose_columns(self, classcount):
        """Fills the final rows as the transpose of the augmented columns"""
        orig_height = self._cooccurrences.shape[0] - classcount
        self._cooccurrences[orig_height:, :-classcount] = \
            self._cooccurrences[:orig_height, -classcount:].T

    def _flip_conditional(self, classcount):
        """Fills the final rows with p(w=i|y=l)"""
        orig_height = self._cooccurrences.shape[0] - classcount
        docwords_csr = self._docwords.tocsr()
        indices = docwords_csr.indices
        indptr = docwords_csr.indptr
        data = docwords_csr.data
        # calculate the label totals
        label_totals = [0 for i in range(classcount)]
        for j in range(orig_height):
            for docnum, data in zip(indices[indptr[j]:indptr[j+1]],
                                    data[indptr[j]:indptr[j+1]]):
                label = self.classorder[self.labels[self.titles[docnum]]]
                label_totals[label] += 1
                # Calculate occurrences for bottom rows
                self._cooccurrences[orig_height+label, j] += 1
        for label, total in enumerate(label_totals):
            self._cooccurrences[orig_height+label, :-classcount] /= total



"""Models for use in ClassTM"""
import numpy as np

import ankura.pipeline
import classtm.labeled


#pylint:disable-msg=too-few-public-methods
class FreeClassifier:
    """Classifier that came for free as part of training FreeClassifyingAnchor"""

    def __init__(self, weights, classorder):
        """Initialize FreeClassifier instance with all the information it needs
            * weights :: 2D np.array
                one row for each class, signifying the linear combination of
                topics that represent a class
            * classorder :: {'class': int}
                dictionary of class names that are mapped to corresponding index
                in weights
        """
        self.weights = weights
        self.classorder = classorder
        self.orderedclasses = [''] * len(self.classorder)
        for cla in self.classorder:
            self.orderedclasses[self.classorder[cla]] = cla

    def predict(self, features):
        """Predict class label for features"""
        bestpos = 0
        bestscore = np.dot(self.weights[0], features[0])
        for i, classweights in enumerate(self.weights[1:]):
            curscore = np.dot(classweights, features[0])
            if curscore > bestscore:
                bestpos = i + 1
                bestscore = curscore
        return self.orderedclasses[bestpos]


def build_train_set(dataset, train_doc_ids, knownresp):
    """Build training set

    Return training set, vocab conversion table, and title ids for training set
    """
    tmp = ankura.pipeline.Dataset(dataset.docwords[:, train_doc_ids],
                                  dataset.vocab,
                                  [dataset.titles[tid] for tid in train_doc_ids])
    corpus_to_train_vocab = [-1] * len(dataset.vocab)
    counter = 0
    for i in range(len(dataset.vocab)):
        # keep track of vocabulary left after dropping test set
        if tmp.docwords[i, :].nnz >= 1:
            corpus_to_train_vocab[i] = counter
            counter += 1
    filtered = ankura.pipeline.filter_rarewords(tmp, 1)
    labels = {}
    for doc, resp in zip(train_doc_ids, knownresp):
        labels[dataset.titles[doc]] = resp
    trainingset = classtm.labeled.ClassifiedDataset(filtered,
                                                    labels,
                                                    dataset.classorder)
    return trainingset, corpus_to_train_vocab, list(range(0, len(train_doc_ids)))


def id_cands_maker(classcount):
    """Returns function that returns lest of anchor word candidates"""
    def id_cands(docwords, doc_threshold):
        """Returns list of anchor word candidates
            * docwords :: scipy.sparse.csc
                sparse matrix of word counts per document; shape is (V, D), where V
                is the vocabulary size and D is the number of documents
            * doc_threshold :: int
                number of documents a word must appear in in order to be an anchor
                word candidate
        """
        candidate_anchors = []
        docwords_csr = docwords.tocsr()
        # assuming that docwords[:-classcount] correspond to label pseudo-words
        for i in range(docwords_csr.shape[0] - classcount):
            if docwords_csr[i, :].nnz > doc_threshold:
                candidate_anchors.append(i)
        return candidate_anchors
    return id_cands


#pylint:disable-msg=too-many-instance-attributes
class FreeClassifyingAnchor:
    """Algorithm that produces a model for classification tasks

    As part of the algorithm, a classifier gets trained for free (i.e., the
    features produced by the model are not used to train a separate classifier)
    """

    def __init__(self, rng, numtopics, expgrad_epsilon):
        """FreeClassifyingAnchor requires the following parameters:
            * rng :: random.Random
                a random number generator
            * numtopics :: int
                the number of topics to look for
            * expgrad_epsilon :: float
                epsilon for exponentiated gradient descent
        """
        self.rng = rng
        self.numtopics = numtopics
        self.expgrad_epsilon = expgrad_epsilon
        self.numsamplesperpredictchain = 5
        self.anchors = None
        self.topics = None
        self.corpus_to_train_vocab = None
        self.classorder = None
        self.predictor = None

    def train(self, dataset, train_doc_ids, knownresp):
        """Train model
            * dataset :: classtm.labeled.ClassifiedDataset
                the complete corpus used for experiments
            * train_doc_ids :: [int]
                the documents used for training as index into dataset.titles
            * knownresp :: [?]
                knownresp[x] is the label for train_doc_ids[x]
        """
        trainingset, self.corpus_to_train_vocab, _ = \
            build_train_set(dataset, train_doc_ids, knownresp)
        self.classorder = trainingset.classorder
        pdim = 1000 if trainingset.vocab_size > 1000 else trainingset.vocab_size
        self.anchors = \
            ankura.anchor.gramschmidt_anchors(trainingset,
                                              self.numtopics,
                                              0.015 * len(trainingset.titles),
                                              project_dim=pdim,
                                              id_cands=id_cands_maker(len(self.classorder)))
        # relying on fact that recover_topics goes through all rows of Q, the
        # cooccurrence matrix in trainingset
        # self.topics has shape (vocabsize, numtopics)
        self.topics = ankura.topic.recover_topics(trainingset,
                                                  self.anchors,
                                                  self.expgrad_epsilon)
        classcount = len(trainingset.classorder)
        self.predictor = FreeClassifier(self.topics[-classcount:],
                                        self.classorder)

    def predict(self, tokens):
        """Predict label"""
        docws = self._convert_vocab_space(tokens)
        if len(docws) <= 0:
            return self.rng.choice(self.classorder)
        features = self._predict_topics(docws)
        return self.predictor.predict(features.reshape((1, -1)))

    def _convert_vocab_space(self, tokens):
        """Change vocabulary from corpus space to training set space"""
        result = []
        for token in tokens:
            conversion = self.corpus_to_train_vocab[token]
            if conversion >= 0:
                result.append(conversion)
        return result

    def cleanup(self):
        """Cleans up any resources used by this instance"""
        pass

    def _predict_topics(self, docws):
        """Predict topic mixture for docws"""
        if len(docws) == 0:
            return np.array([1.0/self.numtopics] * self.numtopics)
        result = np.zeros(self.numtopics)
        for _ in range(self.numsamplesperpredictchain):
            counts, _ = ankura.topic.predict_topics(self.topics,
                                                    docws,
                                                    rng=self.rng)
            result += counts
        result /= (len(docws) * self.numsamplesperpredictchain)
        return result

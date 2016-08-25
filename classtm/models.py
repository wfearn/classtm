"""Models for use in ClassTM"""
from sklearn.linear_model import LogisticRegression
import numpy as np

import activetm.tech.anchor
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
        self.orderedclasses = classtm.labeled.orderclasses(self.classorder)

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


def build_train_set(dataset, train_doc_ids, knownresp, trainsettype):
    """Build training set

    Return training set, vocab conversion table, and title ids for training set

        * dataset :: AbstractClassifiedDataset
        * train_doc_ids :: [int]
        * knownresp :: [?]
            knownresp[i] is label for document train_doc_ids[i]
        * trainsettype :: Constructor for AbstractClassifiedDataset
    """
    filtered, corpus_to_train_vocab = \
        activetm.tech.anchor.get_filtered_for_train(dataset,
                                                    train_doc_ids)
    labels = activetm.tech.anchor.get_labels_for_train(dataset,
                                                       train_doc_ids,
                                                       knownresp)
    trainingset = trainsettype(filtered,
                               labels,
                               dataset.classorder)
    return trainingset, corpus_to_train_vocab, list(range(0, len(train_doc_ids)))


def id_cands_maker(classcount, doc_threshold):
    """Returns function that returns list of anchor word candidates
        * classcount :: int
            number of classes; assumes that the last classcount entries of the
            vocabulary are the label pseudo-words
        * doc_threshold :: int
            number of documents a word must appear in in order to be an anchor
            word candidate

    """
    def identify_candidates(docwords):
        """Returns list of anchor word candidates
            * docwords :: scipy.sparse.csc
                sparse matrix of word counts per document; shape is (V, D), where V
                is the vocabulary size and D is the number of documents
        """
        candidate_anchors = []
        docwords_csr = docwords.tocsr()
        # assuming that docwords[:-classcount] correspond to label pseudo-words
        for i in range(docwords_csr.shape[0] - classcount):
            if docwords_csr[i, :].nnz > doc_threshold:
                candidate_anchors.append(i)
        return candidate_anchors
    return identify_candidates


#pylint:disable-msg=too-many-instance-attributes
class AbstractClassifyingAnchor:
    """Base class for classifying anchor words"""

    #pylint:disable-msg=too-many-arguments
    def __init__(self,
                 rng,
                 numtopics,
                 expgrad_epsilon,
                 dataset_ctor,
                 classifier):
        """AbstractClassifyingAnchor requires the following parameters:
            * rng :: random.Random
                a random number generator
            * numtopics :: int
                the number of topics to look for
            * expgrad_epsilon :: float
                epsilon for exponentiated gradient descent
            * dataset_ctor :: Contructor for AbstractClassifiedDataset
            * classifier :: function(AbstractClassifyingAnchor,
                                     AbstractClassifiedDataset,
                                     [?])
                when called, this function returns an sklearn-style classifier
                that has already been trained
        """
        self.rng = rng
        self.numtopics = numtopics
        self.expgrad_epsilon = expgrad_epsilon
        self.dataset_ctor = dataset_ctor
        self.classifier = classifier
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
            build_train_set(dataset,
                            train_doc_ids,
                            knownresp,
                            self.dataset_ctor)
        self.classorder = trainingset.classorder
        pdim = 1000 if trainingset.vocab_size > 1000 else trainingset.vocab_size
        self.anchors = \
            ankura.anchor.gramschmidt_anchors(trainingset,
                                              self.numtopics,
                                              id_cands_maker(len(self.classorder),
                                                             0.015 * len(trainingset.titles)),
                                              project_dim=pdim)
        # relying on fact that recover_topics goes through all rows of Q, the
        # cooccurrence matrix in trainingset
        # self.topics has shape (vocabsize, numtopics)
        self.topics = ankura.topic.recover_topics(trainingset,
                                                  self.anchors,
                                                  self.expgrad_epsilon)
        self.predictor = self.classifier(self, trainingset, knownresp)

    def predict(self, tokens):
        """Predict label"""
        docws = self._convert_vocab_space(tokens)
        if len(docws) <= 0:
            return self.rng.choice(self.classorder)
        features = self.predict_topics(docws)
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

    def predict_topics(self, docws):
        """Predict topic mixture for docws

        Assuming that docws is in trainingset vocabulary space
        """
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


def free_classifier(freeclassifyinganchor, trainingset, _):
    """Builds a trained FreeClassifier"""
    classcount = len(trainingset.classorder)
    return FreeClassifier(freeclassifyinganchor.topics[-classcount:],
                          freeclassifyinganchor.classorder)


#pylint:disable-msg=too-many-instance-attributes
class FreeClassifyingAnchor(AbstractClassifyingAnchor):
    """Algorithm that produces a model for classification tasks

    As part of the algorithm, a classifier gets trained for free (i.e., the
    features produced by the model are not used to train a separate classifier)
    """

    def __init__(self, rng, numtopics, expgrad_epsilon):
        super(FreeClassifyingAnchor, self).__init__(rng,
                                                    numtopics,
                                                    expgrad_epsilon,
                                                    classtm.labeled.ClassifiedDataset,
                                                    free_classifier)


def build_train_adapter(dataset, train_doc_ids, knownresp):
    """Build train set as SupervisedAnchorDataset"""
    return build_train_set(dataset,
                           train_doc_ids,
                           knownresp,
                           classtm.labeled.SupervisedAnchorDataset)


def logistic_regression(logisticanchor, trainingset, knownresp):
    """Builds trained LogisticRegression"""
    result = LogisticRegression()
    features = np.zeros((len(trainingset.titles), logisticanchor.numtopics))
    for i in range(len(trainingset.titles)):
        topic_mixture = logisticanchor.predict_topics(trainingset.doc_tokens(i))
        features[i, :] = topic_mixture
    result.fit(features, np.array(knownresp))
    return result


class LogisticAnchor(AbstractClassifyingAnchor):
    """Algorithm that produces a model for classification tasks

    This should run as per Nguyen et al. (NAACL 2015), with logistic regression
    """

    def __init__(self, rng, numtopics, expgrad_epsilon):
        super(LogisticAnchor, self).__init__(rng,
                                             numtopics,
                                             expgrad_epsilon,
                                             classtm.labeled.SupervisedAnchorDataset,
                                             logistic_regression)



FACTORY = {'logistic': LogisticAnchor,
           'free': FreeClassifyingAnchor}


def build(rng, settings):
    """Build model according to settings"""
    numtopics = int(settings['numtopics'])
    expgrad_epsilon = float(settings['expgrad_epsilon'])
    return FACTORY[settings['model']](rng, numtopics, expgrad_epsilon)


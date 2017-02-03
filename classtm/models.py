"""Models for use in ClassTM"""
import os
import subprocess
import json

from sklearn.linear_model import LogisticRegression
import numpy as np
import scipy
from scipy.sparse import csc_matrix

import activetm.tech.anchor
import ankura.pipeline
import classtm.labeled


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
LDA_DIR = os.path.join(FILE_DIR, 'ldac')
LDAC_EXE = os.path.join(LDA_DIR, 'lda')
# these are the settings that Nguyen et al. used
LDAC_SETTINGS = os.path.join(LDA_DIR, 'inf-settings.txt')


class LDAHelper:
    """Here so old data will work, not used"""
    def __init__(self, topics, varname):
        pass

#pylint:disable-msg=too-few-public-methods
class VariationalHelper:
    """Helper to get topic mixtures for documents"""

    def __init__(self, topics, varname):
        """Initialize files necessary to call lda-c

            * topics :: 2D np.array
                should have shape (vocab size, number of topics)
            * varname :: String
                output file name root; note that varname must be less than 86
                characters in length (or else lda-c will do some strange things)
        """
        self.varname = varname
        if len(varname) >= 86:
            raise Exception('Output name prefix is too long: '+self.varname)
        self.datafile = self.varname+'_words.txt'
        self.output = self.varname+'_out'
        self.output_gamma = self.output+'-gamma.dat'
        # .beta file has shape (topics, vocab)
        topicscopy = topics.T.copy()
        # lda-c stores topics in log space
        topicscopy += 0.1e-100
        #pylint:disable-msg=no-member
        topicscopy = np.log(topicscopy)
        np.savetxt(varname+'.beta', topicscopy, fmt='%5.10f')
        with open(varname+'.other', 'w') as ofh:
            ofh.write('num_topics '+str(topicscopy.shape[0])+'\n')
            ofh.write('num_terms '+str(topicscopy.shape[1])+'\n')
            # Nguyen et al. use an alpha of 0.1:
            # anchor_python/scripts/create_other_ldac.py
            ofh.write('alpha 0.1\n')

    def predict_topics(self, docwses):
        """Call on lda-c to get gammas

            * docwses :: [[int]]
                the first dimension separates documents; the second dimension
                separates tokens
        Assuming that all documents in docwses are non-empty
        """
        countses = []
        for docws in docwses:
            countses.append(np.bincount(docws))
        with open(self.datafile, 'w') as ofh:
            for counts in countses:
                line = []
                for i, count in enumerate(counts):
                    if count > 0:
                        line.append(str(i)+':'+str(count))
                line.insert(0, str(len(line)))
                ofh.write(' '.join(line)+'\n')
        subprocess.run(
            [
                LDAC_EXE,
                'inf',
                LDAC_SETTINGS,
                self.varname,
                self.datafile,
                self.output])
        return np.loadtxt(self.output_gamma)


class SamplingHelper:
    """Helper to get topic mixtures for documents"""

    def __init__(self, topics, varname):
        """Initialize variables necessary to call ankura

            * topics :: 2D np.array
                should have shape (vocab size, number of topics)
            * varname :: String
                output file name root; note that varname must be less than 86
                characters in length (or else lda-c will do some strange things)
        """
        self.topics = topics
        self.varname = varname
        self.numsamplesperpredictchain = 5

    def predict_topics(self, docwses):
        """Call ankura to get topic mixes for all the documents
            * docwses :: [[int]]
                the first dimension separates documents; the second dimension
                separates tokens
        Assumes that all documents in docwses are non-empty
        """
        numtopics = self.topics.shape[1]
        topic_mixes = np.zeros((len(docwses), numtopics))
        for i, docws in enumerate(docwses):
            result = np.zeros(numtopics)
            for _ in range(self.numsamplesperpredictchain):
                counts, _ = ankura.topic.predict_topics(self.topics,
                                                        docws)
                result += counts
            result /= (len(docws) * self.numsamplesperpredictchain)
            topic_mixes[i,:] = result
        return topic_mixes


#pylint:disable-msg=too-few-public-methods
class FreeClassifier:
    """Classifier that came for free as part of training FreeClassifyingAnchor"""

    def __init__(self, weights, class_given_word, classorder):
        """Initialize FreeClassifier instance with all the information it needs
            * weights :: 2D np.array
                one row for each class, signifying the linear combination of
                topics that represent a class
            * class_given_word :: 2D np.array
                expected shape is (number of classes, vocab size), this is the
                probability of each class given each word
            * classorder :: {'class': int}
                dictionary of class names that are mapped to corresponding index
                in weights
        """
        # column normalize
        epsilon = 1e-7
        modified_weights = weights + epsilon
        column_sums = modified_weights.sum(axis=0)
        self.weights = weights / column_sums
        self.classorder = classorder
        self.orderedclasses = classtm.labeled.orderclasses(self.classorder)
        # Added by Connor to get word features working
        self.class_given_word = class_given_word

    def predict(self, features, doc_words):
        """Predict class labels for each instance in features

            * features :: 2D np.array
                has shape (number of instances, topic count)
            * doc_words :: 2D scipy.sparse.csc_matrix
                has shape (vocab size, number of documents)
        """
        # dot product calculates score for each label for each instance, where
        # labels are lined up along the rows and instances are lined up along
        # the columns
        topic_score = np.dot(self.weights, features.T)
        topic_score = topic_score / topic_score.sum(axis=0)
        word_score = csc_matrix.dot(self.class_given_word, doc_words)
        word_score_sum = word_score.sum(axis=0)
        word_score = word_score / word_score_sum
        score = topic_score + word_score
        # axis tells argmax to choose the highest row per column
        predictions = np.argmax(score, axis=0)
        return np.array([self.orderedclasses[pred] for pred in predictions])


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
        self.vocabsize = None
        self.classorder = None
        self.lda = None
        self.predictor = None

    def train(self, dataset, train_doc_ids, knownresp, varname, lda_helper, anchors_file):
        """Train model
            * dataset :: classtm.labeled.ClassifiedDataset
                the complete corpus used for experiments
            * train_doc_ids :: [int]
                the documents used for training as index into dataset.titles
            * knownresp :: [?]
                knownresp[x] is the label for train_doc_ids[x]
            * varname :: String
                output file name for calling lda-c (important so that parallel
                processes don't stomp on each other)
            * lda_helper :: Class
                used to make an LDA helper (either VariationalHelper or
                SamplingHelper)
            * anchors_file :: String
                name of the file containing the anchors this model should use
                or None if gram-schmidt anchors should be used
        """
        trainingset, self.corpus_to_train_vocab, _ = \
            build_train_set(dataset,
                            train_doc_ids,
                            knownresp,
                            self.dataset_ctor)
        self.vocabsize = trainingset.vocab_size
        self.classorder = trainingset.classorder
        pdim = 1000 if trainingset.vocab_size > 1000 else trainingset.vocab_size
        if anchors_file is None:
            self.anchors = \
                ankura.anchor.gramschmidt_anchors(trainingset,
                                                  self.numtopics,
                                                  id_cands_maker(len(self.classorder),
                                                                 0.015 * len(trainingset.titles)),
                                                  project_dim=pdim)
        else:
            # pull user-made anchors from a JSON file of anchors
            user_file = json.load(open(anchors_file, 'r'))
            # we only want the last group of anchors that were chosen
            user_anchors = user_file[len(user_file)-1]['anchors']
            self.anchors = ankura.anchor.multiword_anchors(trainingset, user_anchors)
            # numtopics is determined at runtime when using user anchors
            self.numtopics = len(self.anchors)
        # relying on fact that recover_topics goes through all rows of Q, the
        # cooccurrence matrix in trainingset
        # self.topics has shape (vocabsize, numtopics)
        self.topics = ankura.topic.recover_topics(trainingset,
                                                  self.anchors,
                                                  self.expgrad_epsilon)
        self.lda = lda_helper(self.topics, varname)
        self.predictor = self.classifier(self, trainingset, knownresp)

    def predict(self, tokenses):
        """Predict labels"""
        docwses = []
        for tokens in tokenses:
            docwses.append(self._convert_vocab_space(tokens))
        features = self.predict_topics(docwses)
        return self.predictor.predict(features)

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

    def predict_topics(self, docwses):
        """Predict topic mixtures for docwses

            * docwses :: [[int]]
                the first dimension separates documents; the second dimension
                separates tokens
        Assuming that docwses is in trainingset vocabulary space
        """
        passon = []
        empties = []
        for i, docws in enumerate(docwses):
            length = len(docws)
            if length > 0:
                passon.append(docws)
            else:
                empties.append(i)
        empty_mix = np.array([1.0/self.numtopics] * self.numtopics)
        topic_mixes = self.lda.predict_topics(passon)
        result = np.zeros((len(docwses), self.numtopics))
        added = 0
        for i in range(len(docwses)):
            if len(empties) > 0 and added < len(empties) and i == empties[added]:
                result[i:] = empty_mix
                added += 1
            else:
                result[i:] = topic_mixes[i-added]
        return result


def free_classifier(freeclassifyinganchor, trainingset, _):
    """Builds a trained FreeClassifier"""
    classcount = len(trainingset.classorder)
    class_topic_weights = freeclassifyinganchor.topics[-classcount:]
    class_given_word = trainingset.Q[:-classcount, -classcount:].T
    return FreeClassifier(class_topic_weights, class_given_word,
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

    def predict(self, tokenses):
        """Predict labels"""
        docwses = []
        doc_words = scipy.sparse.dok_matrix((self.vocabsize-len(self.classorder), len(tokenses)))
        for i, tokens in enumerate(tokenses):
            real_vocab = self._convert_vocab_space(tokens)
            docwses.append(real_vocab)
            for token in real_vocab:
                doc_words[token, i] += 1
        features = self.predict_topics(docwses)
        return self.predictor.predict(features, doc_words.tocsc())


def build_train_adapter(dataset, train_doc_ids, knownresp):
    """Build train set as SupervisedAnchorDataset"""
    return build_train_set(dataset,
                           train_doc_ids,
                           knownresp,
                           classtm.labeled.SupervisedAnchorDataset)


def logistic_regression(logisticanchor, trainingset, knownresp):
    """Builds trained LogisticRegression"""
    docwses = []
    for i in range(len(trainingset.titles)):
        docwses.append(trainingset.doc_tokens(i))
    result = LogisticRegression()
    features = logisticanchor.predict_topics(docwses)
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


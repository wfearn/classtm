"""Models for use in ClassTM"""
import datetime
import os
import subprocess
import json
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import numpy as np
import scipy
from scipy.sparse import csc_matrix

import activetm.tech.anchor
import ankura.pipeline
import classtm.labeled
import classtm.classifier


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
LDA_DIR = os.path.join(FILE_DIR, 'ldac')
LDAC_EXE = os.path.join(LDA_DIR, 'lda')
# these are the settings that Nguyen et al. used
LDAC_SETTINGS = os.path.join(LDA_DIR, 'inf-settings.txt')


def count_tokens(tokens):
    """Count token types found in tokens

        * tokens :: [str]
    """
    result = {}
    for token in tokens:
        if token in result:
            result[token] += 1
        else:
            result[token] = 1
    return result


# pylint:disable-msg=too-few-public-methods
class VariationalHelper:
    """Helper to get topic mixtures for documents"""

    def __init__(self, topics, varname):
        """Initialize files necessary to call lda-c

            * topics :: 2D np.array
                should have shape (vocab size, number of topics)
            * varname :: String
                output file name root; note that varname must be less than 86
                characters in length (or else lda-c will do some strange
                things)
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
        # pylint:disable=no-member
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
            countses.append(count_tokens(docws))
        with open(self.datafile, 'w') as ofh:
            for counts in countses:
                line = []
                line.append(str(len(counts)))
                for token, count in sorted(counts.items()):
                    line.append(str(token)+':'+str(count))
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
                characters in length (or else lda-c will do some strange
                things)
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
            topic_mixes[i, :] = result
        return topic_mixes


# pylint:disable-msg=too-few-public-methods
class FreeClassifier:
    """Classifier that came for free as part of training FreeClassifyingAnchor
    """

    def __init__(self, weights, class_given_word, classorder):
        """Initialize FreeClassifier instance with all the information it needs
            * weights :: 2D np.array
                one row for each class, signifying the linear combination of
                topics that represent a class
            * class_given_word :: 2D np.array
                expected shape is (number of classes, vocab size), this is the
                probability of each class given each word
            * classorder :: {'class': int}
                dictionary of class names that are mapped to corresponding
                index in weights
        """
        # column normalize
        epsilon = 1e-7
        modified_weights = weights + epsilon
        column_sums = modified_weights.sum(axis=0)
        self.weights = modified_weights / column_sums
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
        # wherever sums are 0, make no-op division
        word_score_sum[word_score_sum == 0] = 1
        word_score = word_score / word_score_sum
        score = topic_score + word_score
        # axis tells argmax to choose the highest row per column
        predictions = np.argmax(score, axis=0)
        return np.array([self.orderedclasses[pred] for pred in predictions])


def _get_train_intermediates(dataset, train_doc_ids, knownresp):
    """Gets intermediate data structures for build_*train_set_builder _inner
    functions

        * dataset :: AbstractClassifiedDataset
        * train_doc_ids :: [int]
        * knownresp :: [?]
            knownresp[i] is label for document train_doc_ids[i]
    """
    filtered, corpus_to_train_vocab = \
        activetm.tech.anchor.get_filtered_for_train(dataset,
                                                    train_doc_ids)
    labels = activetm.tech.anchor.get_labels_for_train(dataset,
                                                       train_doc_ids,
                                                       knownresp)
    return filtered, corpus_to_train_vocab, labels


class AbstractTrainSetBuilder(object):
    """Builder for training set

    Originally, I (Nozomu) had written the *TrainSetBuilder code as a few
    closures.  Unfortunately, Python couldn't pickle the closures (the models,
    which keep a reference to how to build training sets, get pickled in the
    experiment code to keep track of data used for plotting), so I
    reimplemented the closures in object-oriented form.
    """

    def __init__(self, dataset_ctor):
        self.dataset_ctor = dataset_ctor

    def build_train_set(self, dataset, train_doc_ids, knownresp):
        """Builds training set with corresponding corpus to training set
        vocabulary
            * train_doc_ids :: [int]
            * knownresp :: [?]
                knownresp[i] is label for document train_doc_ids[i]
            * trainsettype :: SupervisedAnchorDataset constructor
        """
        raise NotImplementedError(
            'Cannot build training set from AbstractTrainSetBuilder')


class TrainSetBuilder(AbstractTrainSetBuilder):
    """Training set builder for SupervisedAnchorDataset"""

    def __init__(self, dataset_ctor):
        """
            * dataset_ctor :: SupervisedAnchorDataset contructor
        """
        super(TrainSetBuilder, self).__init__(dataset_ctor)

    def build_train_set(self, dataset, train_doc_ids, knownresp):
        filtered, corpus_to_train_vocab, labels = _get_train_intermediates(
            dataset,
            train_doc_ids,
            knownresp)
        trainingset = self.dataset_ctor(filtered,
                                        labels,
                                        dataset.classorder)
        return trainingset, corpus_to_train_vocab


class ParameterizedTrainSetBuilder(AbstractTrainSetBuilder):
    """Training set builder for AbstractParameterizedClassifyingDataset"""

    def __init__(self, dataset_ctor, smoothing, label_weight):
        """
            * dataset_ctor ::
                    AbstractParameterizedClassifyingDataset constructor
            * smoothing :: float
                smoothing value used in place of zero for class values in
                unlabeled documents
            * label_weight :: str
                formula for calculating value to place in true class label for
                labeled document; see labeled.get_label_weight_function for
                more details on the form of the formula
        """
        super(ParameterizedTrainSetBuilder, self).__init__(dataset_ctor)
        self.smoothing = smoothing
        self.label_weight = label_weight

    def build_train_set(self, dataset, train_doc_ids, knownresp):
        filtered, corpus_to_train_vocab, labels = _get_train_intermediates(
            dataset,
            train_doc_ids,
            knownresp)
        trainingset = self.dataset_ctor(filtered,
                                        labels,
                                        dataset.classorder,
                                        self.smoothing,
                                        self.label_weight)
        return trainingset, corpus_to_train_vocab


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
                sparse matrix of word counts per document; shape is (V, D),
                where V is the vocabulary size and D is the number of documents
        """
        candidate_anchors = []
        docwords_csr = docwords.tocsr()
        # assuming that docwords[:-classcount] correspond to label pseudo-words
        for i in range(docwords_csr.shape[0] - classcount):
            if docwords_csr[i, :].nnz > doc_threshold:
                candidate_anchors.append(i)
        return candidate_anchors
    return identify_candidates


# pylint:disable-msg=too-many-instance-attributes
class AbstractClassifyingAnchor:
    """Base class for classifying anchor words"""

    # pylint:disable-msg=too-many-arguments
    def __init__(self,
                 rng,
                 numtopics,
                 expgrad_epsilon,
                 train_set_builder,
                 classifier):
        """AbstractClassifyingAnchor requires the following parameters:
            * rng :: random.Random
                a random number generator
            * numtopics :: int
                the number of topics to look for
            * expgrad_epsilon :: float
                epsilon for exponentiated gradient descent
            * build_train_set_f :: AbstractTrainSetBuilder
                an instance of an AbstractTrainSetBuilder that will be used to
                build the training set
            * classifier :: function(AbstractClassifyingAnchor,
                                     AbstractClassifiedDataset,
                                     [?])
                when called, this function returns an sklearn-style classifier
                that has already been trained
        """
        self.rng = rng
        self.numtopics = numtopics
        self.expgrad_epsilon = expgrad_epsilon
        self.train_set_builder = train_set_builder
        self.classifier = classifier
        self.numsamplesperpredictchain = 5
        self.anchors = None
        self.topics = None
        self.corpus_to_train_vocab = None
        self.vocabsize = None
        self.classorder = None
        self.lda = None
        self.predictor = None

    def train(self,
              dataset,
              train_doc_ids,
              knownresp,
              varname,
              lda_helper,
              anchors_file):
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
        trainingset, self.corpus_to_train_vocab = \
            self.train_set_builder.build_train_set(dataset,
                                                   train_doc_ids,
                                                   knownresp)
        self.vocabsize = trainingset.vocab_size
        self.classorder = trainingset.classorder
        pdim = 1000 \
            if trainingset.vocab_size > 1000 else trainingset.vocab_size
        start = time.time()
        if anchors_file is None:
            self.anchors = \
                ankura.anchor.gramschmidt_anchors(
                    trainingset,
                    self.numtopics,
                    id_cands_maker(len(self.classorder),
                                   0.015 * len(trainingset.titles)),
                    project_dim=pdim)
        else:
            # pull user-made anchors from a JSON file of anchors
            user_file = json.load(open(anchors_file, 'r'))
            # we only want the last group of anchors that were chosen
            user_anchors = user_file[len(user_file)-1]['anchors']
            self.anchors = ankura.anchor.multiword_anchors(trainingset,
                                                           user_anchors)
            # numtopics is determined at runtime when using user anchors
            self.numtopics = len(self.anchors)
        # relying on fact that recover_topics goes through all rows of Q, the
        # cooccurrence matrix in trainingset
        # self.topics has shape (vocabsize, numtopics)
        self.topics = ankura.topic.recover_topics(trainingset,
                                                  self.anchors,
                                                  self.expgrad_epsilon)
        end = time.time()
        anchorwords_time = datetime.timedelta(seconds=end-start)
        self.lda = lda_helper(self.topics, varname)
        self.predictor, applytrain_time, train_time = \
            self.classifier(self, trainingset, knownresp)
        return anchorwords_time, applytrain_time, train_time

    def predict(self, tokenses):
        """Predict labels"""
        docwses = []
        for tokens in tokenses:
            docwses.append(self._convert_vocab_space(tokens))
        features = scipy.sparse.hstack(
            [
                self.predict_topics(docwses),
                self.encode(docwses)]).tocsr()
        return self.predictor.predict(features)

    def _convert_vocab_space(self, tokens):
        """Change vocabulary from corpus space to training set space"""
        result = []
        for token in tokens:
            conversion = self.corpus_to_train_vocab[token]
            if conversion >= 0:
                result.append(conversion)
        return result

    def encode(self, docwses):
        """Produces sparse matrix of token counts

        Rows correspond to documents and columns correspond to tokens
        """
        result = scipy.sparse.lil_matrix((len(docwses), self.vocabsize))
        for i, docws in enumerate(docwses):
            for token in sorted(docws):
                result[i, token] += 1
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
            if len(empties) > 0 and \
                    added < len(empties) and \
                    i == empties[added]:
                result[i:] = empty_mix
                added += 1
            else:
                result[i:] = topic_mixes[i-added]
        return result


def free_classifier(freeclassifyinganchor, trainingset, _):
    """Builds a trained FreeClassifier

        * freeclassifyinganchor :: FreeClassifyingAnchor
        * trainingset :: ClassifiedDataset
    """
    classcount = len(trainingset.classorder)
    class_topic_weights = freeclassifyinganchor.topics[-classcount:]
    class_given_word = trainingset.Q[:-classcount, -classcount:].T
    applytrain_time = datetime.timedelta(seconds=0)
    train_time = datetime.timedelta(seconds=0)
    return FreeClassifier(class_topic_weights, class_given_word,
                          freeclassifyinganchor.classorder), \
        applytrain_time, \
        train_time


# pylint:disable-msg=too-many-instance-attributes
class FreeClassifyingAnchor(AbstractClassifyingAnchor):
    """Algorithm that produces a model for classification tasks

    As part of the algorithm, a classifier gets trained for free (i.e., the
    features produced by the model are not used to train a separate classifier)
    """

    def __init__(self,
                 rng,
                 numtopics,
                 expgrad_epsilon,
                 smoothing,
                 label_weight):
        """
            * smoothing :: float
                smoothing value used in place of zero for class values in
                unlabeled documents
            * label_weight :: str
                formula for calculating value to place in true class label for
                labeled document; see labeled.get_label_weight_function for
                more details on the form of the formula
        """
        super(FreeClassifyingAnchor, self).__init__(
            rng,
            numtopics,
            expgrad_epsilon,
            ParameterizedTrainSetBuilder(
                classtm.labeled.ClassifiedDataset,
                smoothing,
                label_weight),
            free_classifier)

    def predict(self, tokenses):
        """Predict labels"""
        docwses = []
        doc_words = scipy.sparse.dok_matrix(
            (self.vocabsize-len(self.classorder), len(tokenses)))
        for i, tokens in enumerate(tokenses):
            real_vocab = self._convert_vocab_space(tokens)
            docwses.append(real_vocab)
            for token in real_vocab:
                doc_words[token, i] += 1
        features = self.predict_topics(docwses)
        return self.predictor.predict(features, doc_words.tocsc())


def sklearn_classifier(anchor, trainingset, knownresp, classifier):
    """Builds trained classifier"""
    start = time.time()
    docwses = []
    for i in range(len(trainingset.titles)):
        docwses.append(trainingset.doc_tokens(i))
    features = scipy.sparse.hstack(
        [
            anchor.predict_topics(docwses),
            anchor.encode(docwses)]).tocsr()
    end = time.time()
    applytrain_time = datetime.timedelta(seconds=end-start)
    start = time.time()
    result = classifier()
    result.fit(features, np.array(knownresp))
    end = time.time()
    train_time = datetime.timedelta(seconds=end-start)
    return result, applytrain_time, train_time


def logistic_regression(logisticanchor, trainingset, knownresp):
    """Builds trained LogisticRegression"""
    return sklearn_classifier(logisticanchor,
                              trainingset,
                              knownresp,
                              LogisticRegression)


def svm(svmanchor, trainingset, knownresp):
    """Builds trained SVC"""
    return sklearn_classifier(svmanchor, trainingset, knownresp, SVC)


def random_forest(rfanchor, trainingset, knownresp):
    """Builds trained RandomForestClassifier"""
    return sklearn_classifier(rfanchor,
                              trainingset,
                              knownresp,
                              RandomForestClassifier)


def naive_bayes(nbanchor, trainingset, knownresp):
    """Builds trained MultinomialNB"""
    return sklearn_classifier(nbanchor, trainingset, knownresp, MultinomialNB)


def incremental_sklearn(anchor, trainingset, classifier):
    """Builds trained classifier for partially labeled corpus"""
    start = time.time()
    docwses = []
    knownresp = []
    for title, label in trainingset.labels.items():
        docwses.append(trainingset.doc_tokens(trainingset.titlesorder[title]))
        knownresp.append(label)
    features = scipy.sparse.hstack(
        [
            anchor.predict_topics(docwses),
            anchor.encode(docwses)]).tocsr()
    end = time.time()
    applytrain_time = datetime.timedelta(seconds=end-start)
    start = time.time()
    result = classifier()
    result.fit(features, np.array(knownresp))
    end = time.time()
    train_time = datetime.timedelta(seconds=end-start)
    return result, applytrain_time, train_time


def incremental_logistic_regression(logisticanchor, trainingset):
    """Builds trained LogisticRegression for partially labeled corpus

        * logisticanchor :: IncrementalLogisticAnchor
        * trainingset :: IncrementalSupervisedAnchorDataset
    """
    return incremental_sklearn(logisticanchor, trainingset, LogisticRegression)


def incremental_svm(svmanchor, trainingset):
    """Builds trained SVC for partially labeled corpus"""
    return incremental_sklearn(svmanchor, trainingset, SVC)


def incremental_random_forest(rfanchor, trainingset):
    """Builds trained RandomForestClassifier for partially labeled corpus"""
    return incremental_sklearn(rfanchor, trainingset, RandomForestClassifier)


def incremental_naive_bayes(nbanchor, trainingset):
    """Builds trained MultinomialNB for partially labeled corpus"""
    return incremental_sklearn(nbanchor, trainingset, MultinomialNB)


def incremental_free_classifier(freeclassifyinganchor, trainingset):
    """Builds trained FreeClassifier"""
    # conveniently, free_classifier already does everything we needed, except
    # it had the wrong number of parameters
    return free_classifier(freeclassifyinganchor, trainingset, None)


def incremental_tsvm(tsvmanchor, trainingset):
    """Builds trained TSVM for partially labeled corpus"""
    start = time.time()
    docwses = []
    knownresp = []
    for title in trainingset.titles:
        docwses.append(trainingset.doc_tokens(trainingset.titlesorder[title]))
        knownresp.append(
            trainingset.labels[title]
            if title in trainingset.labels else 'unknown')
    features = scipy.sparse.hstack(
        [
            tsvmanchor.predict_topics(docwses),
            tsvmanchor.encode(docwses)]).tocsr()
    end = time.time()
    applytrain_time = datetime.timedelta(seconds=end-start)
    start = time.time()
    result = classtm.classifier.TSVM(
        tsvmanchor.lda.varname,
        tsvmanchor.classorder)
    result.fit(features, np.array(knownresp))
    end = time.time()
    train_time = datetime.timedelta(seconds=end-start)
    return result, applytrain_time, train_time


class LogisticAnchor(AbstractClassifyingAnchor):
    """Algorithm that produces a model for classification tasks

    This should run as per Nguyen et al. (NAACL 2015), with logistic regression
    """

    def __init__(self, rng, numtopics, expgrad_epsilon):
        super(LogisticAnchor, self).__init__(
            rng,
            numtopics,
            expgrad_epsilon,
            TrainSetBuilder(classtm.labeled.SupervisedAnchorDataset),
            logistic_regression)


class SVMAnchor(AbstractClassifyingAnchor):
    """Algorithm that produces a model for classification tasks

    This should run as per Nguyen et al. (NAACL 2015), with SVM
    """

    def __init__(self, rng, numtopics, expgrad_epsilon):
        super(SVMAnchor, self).__init__(
            rng,
            numtopics,
            expgrad_epsilon,
            TrainSetBuilder(classtm.labeled.SupervisedAnchorDataset),
            svm)


class RFAnchor(AbstractClassifyingAnchor):
    """Algorithm that produces a model for classification tasks

    This should run as per Nguyen et al. (NAACL 2015), with random forest
    """

    def __init__(self, rng, numtopics, expgrad_epsilon):
        super(RFAnchor, self).__init__(
            rng,
            numtopics,
            expgrad_epsilon,
            TrainSetBuilder(classtm.labeled.SupervisedAnchorDataset),
            random_forest)


class NBAnchor(AbstractClassifyingAnchor):
    """Algorithm that produces a model for classification tasks

    This should run as per Nguyen et al. (NAACL 2015), with naive bayes
    """

    def __init__(self, rng, numtopics, expgrad_epsilon):
        super(NBAnchor, self).__init__(
            rng,
            numtopics,
            expgrad_epsilon,
            TrainSetBuilder(classtm.labeled.SupervisedAnchorDataset),
            naive_bayes)


class AbstractIncrementalAnchor(AbstractClassifyingAnchor):
    """Superclass for anchor words with incrementally labeled corpus

        * self.classifier :: function(AbstractIncrementalAnchor,
                                      AbstractClassifiedDataset)
            when called, this function returns an sklearn-style classifier that
            has already been trained; this self.classifier differs from the one
            in AbstractClassifyingAnchor in that the other one assumes that all
            documents in the dataset will be used in training; this one assumes
            that only labeled data in the dataset will be used in training
    """

    def __init__(self, rng, numtopics, expgrad_epsilon, classifier):
        super(AbstractIncrementalAnchor, self).__init__(rng,
                                                        numtopics,
                                                        expgrad_epsilon,
                                                        None,
                                                        classifier)

    def train(self, dataset, varname, lda_helper, anchors_file):
        """Train model
            * dataset :: classtm.labeled.IncrementalSupervisedAnchorDataset
                the corpus used for experiments
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
        if isinstance(dataset, classtm.labeled.ClassifiedDataset):
            self.corpus_to_train_vocab = list(
                range(len(dataset.origvocabsize)))
        else:
            self.corpus_to_train_vocab = list(range(len(dataset.vocab)))
        trainingset = dataset
        self.vocabsize = trainingset.vocab_size
        self.classorder = trainingset.classorder
        pdim = 1000 \
            if trainingset.vocab_size > 1000 else trainingset.vocab_size
        start = time.time()
        if anchors_file is None:
            # assumes that trainingset.Q has
            # len(self.corpus_to_train_vocab)+len(self.classorder) columns
            self.anchors = \
                ankura.anchor.gramschmidt_anchors(
                    trainingset,
                    self.numtopics,
                    id_cands_maker(len(self.classorder),
                                   0.015 * len(trainingset.titles)),
                    project_dim=pdim)
        else:
            # pull user-made anchors from a JSON file of anchors
            user_file = json.load(open(anchors_file, 'r'))
            # we only want the last group of anchors that were chosen
            user_anchors = user_file[len(user_file)-1]['anchors']
            self.anchors = ankura.anchor.multiword_anchors(trainingset,
                                                           user_anchors)
            # numtopics is determined at runtime when using user anchors
            self.numtopics = len(self.anchors)
        # relying on fact that recover_topics goes through all rows of Q, the
        # cooccurrence matrix in trainingset
        # self.topics has shape (vocabsize, numtopics)
        self.topics = ankura.topic.recover_topics(trainingset,
                                                  self.anchors,
                                                  self.expgrad_epsilon)
        end = time.time()
        anchorwords_time = datetime.timedelta(seconds=end-start)
        self.lda = lda_helper(self.topics, varname)
        self.predictor, applytrain_time, train_time = \
            self.classifier(self, trainingset)
        return anchorwords_time, applytrain_time, train_time


class IncrementalLogisticAnchor(AbstractIncrementalAnchor):
    """LogisticAnchor with incrementally labeled corpus"""

    def __init__(self, rng, numtopics, expgrad_epsilon):
        super(IncrementalLogisticAnchor, self).__init__(
            rng,
            numtopics,
            expgrad_epsilon,
            incremental_logistic_regression)


class IncrementalSVMAnchor(AbstractIncrementalAnchor):
    """SVMAnchor with incrementally labeled corpus"""

    def __init__(self, rng, numtopics, expgrad_epsilon):
        super(IncrementalSVMAnchor, self).__init__(rng,
                                                   numtopics,
                                                   expgrad_epsilon,
                                                   incremental_svm)


class IncrementalRFAnchor(AbstractIncrementalAnchor):
    """RFAnchor with incrementally labeled corpus"""

    def __init__(self, rng, numtopics, expgrad_epsilon):
        super(IncrementalRFAnchor, self).__init__(rng,
                                                  numtopics,
                                                  expgrad_epsilon,
                                                  incremental_random_forest)


class IncrementalNBAnchor(AbstractIncrementalAnchor):
    """RFAnchor with incrementally labeled corpus"""

    def __init__(self, rng, numtopics, expgrad_epsilon):
        super(IncrementalNBAnchor, self).__init__(rng,
                                                  numtopics,
                                                  expgrad_epsilon,
                                                  incremental_naive_bayes)


class IncrementalTSVMAnchor(AbstractIncrementalAnchor):
    """Incrementally labeled corpus with transductive SVM"""

    def __init__(self, rng, numtopics, expgrad_epsilon):
        super(IncrementalTSVMAnchor, self).__init__(rng,
                                                    numtopics,
                                                    expgrad_epsilon,
                                                    incremental_tsvm)


class IncrementalFreeClassifyingAnchor(AbstractIncrementalAnchor):
    """FreeClassifyingAnchor with incrementally labeled corpus"""

    def __init__(self, rng, numtopics, expgrad_epsilon):
        super(IncrementalFreeClassifyingAnchor, self).__init__(
            rng,
            numtopics,
            expgrad_epsilon,
            incremental_free_classifier)

    def predict(self, tokenses):
        """Predict labels"""
        docwses = []
        data = []
        indices = []
        indptr = [0]
        for tokens in tokenses:
            real_vocab = self._convert_vocab_space(tokens)
            docwses.append(real_vocab)
            tmp = count_tokens(real_vocab)
            for token, count in sorted(tmp.items()):
                data.append(count)
                indices.append(token)
            indptr.append(len(data))
        features = self.predict_topics(docwses)
        doc_words = scipy.sparse.csc_matrix(
            (data, indices, indptr),
            shape=(self.vocabsize-len(self.classorder), len(tokenses)))
        return self.predictor.predict(features, doc_words.tocsc())


FACTORY = {'logistic': LogisticAnchor,
           'free': FreeClassifyingAnchor,
           'svm': SVMAnchor,
           'rf': RFAnchor,
           'nb': NBAnchor}


INCFACTORY = {'inclog': [IncrementalLogisticAnchor,
                         classtm.labeled.IncrementalSupervisedAnchorDataset],
              'inclognormed': [IncrementalLogisticAnchor,
                  classtm.labeled.IncrementalSupervisedNormalizedAnchorDataset],
              'incfree': [IncrementalFreeClassifyingAnchor,
                          classtm.labeled.IncrementalClassifiedDataset],
              'quickincfree': [
                  IncrementalFreeClassifyingAnchor,
                  classtm.labeled.QuickIncrementalClassifiedDataset],
              'nonegsfree': [
                  IncrementalFreeClassifyingAnchor,
                  classtm.labeled.ZeroEpsilonDataset],
              'zeronegsfree': [
                  IncrementalFreeClassifyingAnchor,
                  classtm.labeled.ZeroNegativesDataset],
              'projfree': [
                  IncrementalFreeClassifyingAnchor,
                  classtm.labeled.ProjectedDataset],
              'incsvm': [IncrementalSVMAnchor,
                         classtm.labeled.IncrementalSupervisedAnchorDataset],
              'incsvmnormed': [IncrementalSVMAnchor,
                  classtm.labeled.IncrementalSupervisedNormalizedAnchorDataset],
              'incrf': [IncrementalRFAnchor,
                        classtm.labeled.IncrementalSupervisedAnchorDataset],
              'incrfnormed': [IncrementalRFAnchor,
                  classtm.labeled.IncrementalSupervisedNormalizedAnchorDataset],
              'incnb': [IncrementalNBAnchor,
                        classtm.labeled.IncrementalSupervisedAnchorDataset],
              'incnbnormed': [IncrementalNBAnchor,
                  classtm.labeled.IncrementalSupervisedNormalizedAnchorDataset],
              'inctsvm': [IncrementalTSVMAnchor,
                          classtm.labeled.IncrementalSupervisedAnchorDataset],
              'inctsvmnormed': [IncrementalTSVMAnchor,
                  classtm.labeled.IncrementalSupervisedNormalizedAnchorDataset],
              }


def build(rng, settings):
    """Build model according to settings"""
    numtopics = int(settings['numtopics'])
    expgrad_epsilon = float(settings['expgrad_epsilon'])
    if settings['model'] == 'free':
        smoothing = float(settings['smoothing'])
        label_weight = settings['label_weight']
        return FACTORY[settings['model']](rng,
                                          numtopics,
                                          expgrad_epsilon,
                                          smoothing,
                                          label_weight)
    return FACTORY[settings['model']](rng,
                                      numtopics,
                                      expgrad_epsilon)


def initialize(rng, dataset, settings):
    """Build model according to settings"""
    numtopics = int(settings['numtopics'])
    expgrad_epsilon = float(settings['expgrad_epsilon'])
    modeltype, datasettype = INCFACTORY[settings['model']]
    return modeltype(rng, numtopics, expgrad_epsilon),\
        datasettype(dataset, settings)

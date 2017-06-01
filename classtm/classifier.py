"""Classifiers"""
import os
import subprocess

import numpy as np


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
SVM_DIR = os.path.join(FILE_DIR, 'svm_light')
SVM_LEARN = os.path.join(SVM_DIR, 'svm_learn')
SVM_CLASSIFY = os.path.join(SVM_DIR, 'svm_classify')


class TSVM:
    """Transductive support vector machine"""

    def __init__(self, varname, classorder):
        self.outdir = varname+'_tsvm'
        if len(varname) >= 86:
            raise Exception('Output name prefix is too long: '+self.varname)
        os.makedirs(self.outdir, exist_ok=True)
        self.train_prefix = os.path.join(self.outdir, 'train')
        self.model_prefix = os.path.join(self.outdir, 'model')
        self.test_name = os.path.join(self.outdir, 'test.dat')
        self.pred_prefix = os.path.join(self.outdir, 'pred')
        self.classorder = classorder
        self.orderedclasses = [0] * len(self.classorder)
        for key, val in self.classorder.items():
            self.orderedclasses[val] = key

    def _train_name(self, label):
        return self.train_prefix+'_'+str(label)+'.dat'

    def _model_name(self, label):
        return self.model_prefix+'_'+str(label)

    def _pred_name(self, label):
        return self.pred_prefix+'_'+str(label)

    def _write_feats(self, ofh, feats):
        """Writes the features into the data file"""
        # expecting feats to be a csr row
        for col, datum in zip(feats.indices[feats.indptr[0]:feats.indptr[1]],
                              feats.data[feats.indptr[0]:feats.indptr[1]]):
            ofh.write(str(col+1))
            ofh.write(':')
            ofh.write(str(datum))
            ofh.write(' ')

    def fit(self, features, labels):
        """Call SVMLight for transductive SVM training

        features must be a csr matrix
        """
        for label_type in self.classorder:
            train_file = self._train_name(label_type)
            with open(train_file, 'w') as ofh:
                for feats, label in zip(features, labels):
                    if label_type == 'unknown':
                        ofh.write('0 ')
                    elif label_type == label:
                        ofh.write('+1 ')
                    else:
                        ofh.write('-1 ')
                    self._write_feats(ofh, feats)
                    ofh.write('\n')
            subprocess.run(
                [
                    SVM_LEARN,
                    train_file,
                    self._model_name(label_type)])

    def predict(self, features):
        """Call SVMLight for transductive SVM prediction"""
        with open(self.test_name, 'w') as ofh:
            for feats in features:
                ofh.write('0 ')
                self._write_feats(ofh, feats)
                ofh.write('\n')
        predictions = []
        for label_type in self.classorder:
            pred_name = self._pred_name(label_type)
            subprocess.run(
                [
                    SVM_CLASSIFY,
                    self.test_name,
                    self._model_name(label_type),
                    pred_name])
            tmp = []
            with open(pred_name) as ifh:
                for line in ifh:
                    line = line.strip()
                    tmp.append(float(line))
            predictions.append(tmp)
        predictions = np.argmax(np.array(predictions), axis=0)
        return np.array([self.orderedclasses[a] for a in predictions])

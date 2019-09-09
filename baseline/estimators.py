import sys
sys.path.append("..")

import lib.notebook_helpers as nh
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import preprocessing
import pandas as pd
import numpy as np
import implicit
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, Normalizer



class TemplateEstimator(BaseEstimator, TransformerMixin):
    """Estimator base class.
    Follows the sklearn estimator API.
    """
    def __init__(self, n=3):
        self.n = n

    def fit(self, X=None, y=None):
        """Makes sure that the input is of the right type.
        """
        for input_ in [(X, "X"), (y, "y")]:
            if input_[0] is not None and type(input_[0]) is not sp.csr_matrix:
                print("Warning: input \"" + input_[1] + "\" is not of csr_matrix type.")

    def predict(self, X=None, start_index=0, n=None):
        """Returns the top-n predictions as a one-hot multi-element array.
        """
        predictions = self.predict_score(X, start_index=start_index)

        if n is None:
            n = self.n
        tops = nh.get_top_n_tags(predictions, n=n)
        return nh.from_keras_format(list(map(lambda x: x + 1, tops)), predictions.shape[1])

    def predict_score(self, X, start_index):
        """Returns the probabilities of the classes. If the probabilities
        have been calculated already, returns the calculated value.
        """

        raise NotImplementedError("Please implement this method")

class ALSEstimator(TemplateEstimator):
    """Models items (dim 1) as items and tags (dim 2) as users.
    """
    def __init__(self, factors=50,
                       regularization=0.01,
                       iterations=15,
                       filter_seen=True,
                       show_progress=True,
                       n=3):
        super().__init__(n)
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.filter_seen = filter_seen
        self.show_progress = show_progress

    def fit(self, X=None, y=None):
        super().fit(None, y)
        self._model = implicit.als.AlternatingLeastSquares(factors=self.factors,
                                             regularization=self.regularization,
                                             iterations=self.iterations,
                                             dtype=np.float32,
                                             use_native=True)
        try:
            self._model.fit(y, show_progress=self.show_progress)
        except AttributeError:
            self._model.fit(sp.csc_matrix(y), show_progress=self.show_progress)

        if self.filter_seen:
            self._fit_y = y
        self._fitted = True
        return self

    def predict_score(self, X=None, start_index=0):
        check_is_fitted(self, ['_fitted'])
        preds = np.dot(self._model.item_factors, self._model.user_factors.T)
        if self.filter_seen:
            preds[self._fit_y.nonzero()] = -99
        if X is not None:
            preds = preds[start_index: start_index + X.shape[0]]
        return preds

class NaiveBayesEstimator(TemplateEstimator):
    def __init__(self, alpha=1, n=3):
        super().__init__(n)
        self.alpha = alpha

    def fit(self, X=None, y=None):
        super().fit(X, y)
        self._model = OneVsRestClassifier(MultinomialNB(alpha=self.alpha))
        self._model.fit(X, y)

        self._fitted = True

        return self

    def predict_score(self, X, start_index=None):
        check_is_fitted(self, ['_fitted'])

        return self._model.predict_proba(X)

class SVMEstimator(TemplateEstimator):
    def __init__(self, n=3, C=10):
        self.n = n
        self.C = C

    def fit(self, X=None, y=None):
        super().fit(X, y)
        self.transformer_ = Normalizer()
        X_transformed = self.transformer_.fit_transform(X)

        self._model = OneVsRestClassifier(LinearSVC(C=self.C))
        self._model.fit(X_transformed, y)

        self._fitted = True

        return self

    def predict_score(self, X, start_index=None):
        check_is_fitted(self, ['_fitted'])

        X_transformed = self.transformer_.transform(X)
        d_score = self._model.decision_function(X_transformed)
        # scale to 0 - 1 interval
        return (d_score - np.min(d_score)/np.ptp(d_score))

class BaselineModel(BaseEstimator):
    def __init__(self,
                    factors=50,
                    regularization=0.01,
                    iterations=15,
                    filter_seen=True,
                    show_progress=True,
                    n=3,
                    content_scale_factor=0.2,
                    alpha=4):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.filter_seen = filter_seen
        self.show_progress = show_progress
        self.n = n
        self.content_scale_factor = content_scale_factor
        self.alpha = alpha

    def fit(self, X, y):
        self.cf = ALSEstimator(self.factors,
                               self.regularization,
                               self.iterations,
                               self.filter_seen,
                               self.show_progress,
                               self.n)
        self.content = NaiveBayesEstimator(self.alpha,
                                           self.n)


        self.cf = self.cf.fit(y=y)
        self.content = self.content.fit(X, y)

        self._fitted = True
        return self

    def predict(self, X=None, y=None, start_index=0, n=None):
        check_is_fitted(self, ['_fitted'])

        if n is None:
            n = self.n

        cf_predictions = self.cf.predict_score(X, start_index=start_index)
        cf_top_n_scores = self._top_n_scores(cf_predictions, n)

        content_predictions = self.content.predict_score(X) * self.content_scale_factor  # Use of scale_factor hyperparameter
        content_top_n_scores = self._top_n_scores(content_predictions, n)

        def best_n(cf_cands, content_cands, n):
            best_n = sorted(cf_cands + content_cands, key = lambda x: x[1], reverse=True)

            # Don't add tuples with the same index
            retlist = []
            seen_indexes = set()
            for cand in best_n:
                if cand[0] not in seen_indexes:
                    retlist.append(cand)
                    seen_indexes.add(cand[0])
            return retlist[:n]

        predictions = [
            best_n(cf_cands, content_cands, n)
            for cf_cands, content_cands in zip(cf_top_n_scores, content_top_n_scores)
        ]

        top_indexes = np.array([[index for index, _ in top_list] for top_list in predictions])

        return nh.from_keras_format(list(map(lambda x: x + 1, top_indexes)), cf_predictions.shape[1])

    def _top_n_scores(self, score_matrix, n):
        """Returns lists of tuples with top-n element indexes and the
        scores of them, given the score matrix.
        """
        top_n_indexes_lists = nh.get_top_n_tags(score_matrix, n=n)

        f = lambda x: x[x.argsort()[-n:][::-1]]
        top_n_scores_lists = np.apply_along_axis(f, 1, score_matrix)
        return [list(zip(i_list, p_list)) for i_list, p_list in zip(top_n_indexes_lists, top_n_scores_lists)]

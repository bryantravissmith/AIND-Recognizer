import math
import statistics
import sys
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, words: dict, hwords: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=None, verbose=False):
        self.words = words
        self.hwords = hwords
        self.sequences = words[this_word]
        self.X, self.lengths = hwords[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        x = []
        y = []

        if len(self.sequences) > 1:
            for n in range(self.min_n_components, self.max_n_components):
                split_method = KFold(n_splits=2) #KFold(n_splits=min(len(self.sequences), 5))
                count = 0
                bic = 0
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    try:
                        X = np.concatenate(np.array(self.sequences)[cv_train_idx])
                        l = np.array(self.lengths)[cv_train_idx]
                        model = GaussianHMM(n_components=n,
                                            covariance_type="diag",
                                            n_iter=1000,
                                            random_state=121,
                                            verbose=False)\
                            .fit(X, l)

                        X_test = np.concatenate(np.array(self.sequences)[cv_test_idx])
                        l_test = np.array(self.lengths)[cv_test_idx]
                        bic += -2 * model.score(X_test, l_test) + 2 * n * len(self.sequences[0]) * np.log(np.array(self.lengths)[cv_test_idx].sum())
                        count += 1
                    except:
                        pass

                if count > 0 and bic != 0:
                    y.append(bic / count)
                    x.append(n)

        if len(y) > 0:
            n_best = x[y.index(min(y))]
        else:
            n_best = 2

        hmm_model = GaussianHMM(n_components=n_best, covariance_type="diag", n_iter=1000,
                                random_state=self.random_state, verbose=False).fit(self.X, self.lengths)

        return hmm_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        x = []
        y = []

        if len(self.sequences) > 1:
            for n in range(self.min_n_components, self.max_n_components):
                split_method = KFold(n_splits=2) #KFold(n_splits=min(len(self.sequences), 5))
                count = 0
                logl = 0
                dic = 0
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    try:
                         X = np.concatenate(np.array(self.sequences)[cv_train_idx])
                        l = np.array(self.lengths)[cv_train_idx]
                        model = GaussianHMM(n_components=n,
                                            covariance_type="diag",
                                            n_iter=1000,
                                            random_state=121,
                                            verbose=False)\
                            .fit(X, l)

                        X_test = np.concatenate(np.array(self.sequences)[cv_test_idx])
                        l_test = np.array(self.lengths)[cv_test_idx]
                        logl = model.score(X_test, l_test)

                        count_dl = 0
                        log_dl = 0
                        correction = 0

                        for word in self.words.keys():
                            if word != self.this_word:
                                log_dl += model.score(self.hwords[word][0], self.hwords[word][1])
                                count_dl += 1
                                correction += 2 * n * len(self.sequences[0]) / 2 * np.log(sum(self.hwords[word][1]) / np.array(self.lengths)[cv_test_idx].sum())

                        dic += logl - log_dl / (count_dl - 1) + correction
                        count += 1
                    except:
                        #print(sys.exc_info())
                        pass

                if count > 0 and dic != 0:
                    y.append(dic / count)
                    x.append(n)

        if len(y) > 0:
            n_best = x[y.index(max(y))]
        else:
            n_best = 2
        hmm_model = GaussianHMM(n_components=n_best, covariance_type="diag", n_iter=1000,
                                random_state=self.random_state, verbose=False).fit(self.X, self.lengths)

        return hmm_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        x = []
        y = []
        if len(self.sequences) > 1:
            for n in range(self.min_n_components, self.max_n_components):
                split_method = KFold(n_splits=2) #KFold(n_splits=min(len(self.sequences), 5))
                count = 0
                logl = 0
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    try:
                        X = np.concatenate(np.array(self.sequences)[cv_train_idx])
                        l = np.array(self.lengths)[cv_train_idx]
                        model = GaussianHMM(n_components=n,
                                            covariance_type="diag",
                                            n_iter=1000,
                                            random_state=121,
                                            verbose=False)\
                            .fit(X, l)

                        X_test = np.concatenate(np.array(self.sequences)[cv_test_idx])
                        l_test = np.array(self.lengths)[cv_test_idx]
                        logl += model.score(X_test, l_test)
                        count += 1
                    except:
                        pass

                if count > 0 and logl != 0:
                    y.append(logl / count)
                    x.append(n)

        if len(y) > 0:
            n_best = x[y.index(max(y))]
        else:
            n_best = 2
        hmm_model = GaussianHMM(n_components=n_best, covariance_type="diag", n_iter=1000,
                                random_state=self.random_state, verbose=False).fit(self.X, self.lengths)

        return hmm_model
        #raise NotImplementedError

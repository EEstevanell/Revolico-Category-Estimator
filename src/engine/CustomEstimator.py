from .base import BaseAdvertiseClassifier, movie_reviews_test, load_movie_reviews
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import numpy as np

class CustomEstimator(BaseAdvertiseClassifier):
    def __init__(self, language = 'english', vectorizer = 'tf_vectorizer', penalty = 'l2'):
        super(type(self), self).__init__(language = language, vectorizer = vectorizer)
        self.classifier = MLPClassifier(hidden_layer_sizes=[27] * 2, activation='tanh')

    def _semantic(self, X):
        features = []
        # for sentence in X:
        #     counter = Counter()
        #     for token in sentence:
        #         counter[token.pos_] += 1
        #         counter[token.tag_] += 1
        #         counter[token.dep_] += 1
        #     features.append(counter)

        # self.dv = DictVectorizer()
        # return self.dv.fit_transform(features)
        return features

    def transform(self, X):
        X_append = self._semantic(X)
        new_X = self.fitted_vectorizer.transform(X)
        imputed_X = self.imputer.transform(new_X)
        return imputed_X

    def fit(self, X, y):
        if self.classifier is None:
            raise NotImplementedError("need to provide an estimator for BaseAdvertiseClassifier class")

        fit_vectorizer = self.__getattribute__(self.vectorizer)
        fit_vectorizer(X)

        X_append = self._semantic(X)
        new_X = self.fitted_vectorizer.transform(X)
        self.imputer = SimpleImputer(strategy="median")
        imputed_X = self.imputer.fit_transform(new_X)
        return self.classifier.fit(imputed_X, y)

# X, y = load_movie_reviews()
# movie_reviews_test(CustomEstimator())
# es = CustomEstimator()
# X_train, X_test, y_train, y_test = train_test_split( X, y, shuffle = True)

# es.fit(X_train, y_train)

# results = es.score(X_test, y_test)
# print(results)
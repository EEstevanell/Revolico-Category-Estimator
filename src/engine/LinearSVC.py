from .base import BaseAdvertiseClassifier, movie_reviews_test
from sklearn.svm import LinearSVC

class LinearSVClassifier(BaseAdvertiseClassifier):
    """
    Multiclass classifier based on Support Vector Machine
    """
    def __init__(self, language = 'english', vectorizer = 'tf_vectorizer', penalty = 'l2'):
        super(type(self), self).__init__(language = language, vectorizer = vectorizer)
        self.penalty = penalty
        self.classifier = LinearSVC()

    def get_params(self, deep = True):
        params = {}
        params.update(super().get_params(deep))
        return params

# movie_reviews_test(LinearSVClassifier())

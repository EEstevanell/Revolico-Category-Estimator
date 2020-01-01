from .base import BaseAdvertiseClassifier, movie_reviews_test
from sklearn.linear_model import LogisticRegression

class LogisticRegressionClassifier(BaseAdvertiseClassifier):
    """
    Multiclass classifier based on Logistic regression
    """
    def __init__(self, language = 'english', vectorizer = 'tf_vectorizer', penalty = 'l2'):
        super(type(self), self).__init__(language = language, vectorizer = vectorizer)
        self.penalty = penalty
        self.classifier = LogisticRegression(multi_class = 'multinomial', penalty = penalty, solver = 'newton-cg')

    def get_params(self, deep = True):
        params = {'penalty': self.penalty}
        params.update(super().get_params(deep))
        return params


# movie_reviews_test(LogisticRegressionClassifier())
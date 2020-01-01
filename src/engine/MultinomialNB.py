from .base import BaseAdvertiseClassifier, movie_reviews_test
from sklearn.naive_bayes import MultinomialNB

class MultinomialNBClassifier(BaseAdvertiseClassifier):
    """
    Multiclass classifier based on Multinomial Naive Bayes
    """
    def __init__(self, language = 'english', vectorizer = 'tf_vectorizer', alpha = 1.0):
        super(type(self), self).__init__(language = language, vectorizer = vectorizer)
        self.alpha = alpha
        self.classifier = MultinomialNB(alpha = alpha)

    def get_params(self, deep = True):
        params = {'alpha': self.alpha}
        params.update(super().get_params(deep))
        return params

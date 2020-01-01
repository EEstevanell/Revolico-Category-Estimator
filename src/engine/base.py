from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection._split import KFold
from nltk.corpus.reader.wordlist import WordListCorpusReader
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize, WordPunctTokenizer, RegexpTokenizer
from nltk.corpus import wordnet
import random

class BaseAdvertiseClassifier(BaseEstimator):
    """
    Base class for advertise classifier. Inherited classes must provide an attribute
    `classifier` with an estimator that implements a `fit` function and a `predict` function.

    `classifier` must be a multiclass classifier and must be setted after initializing the `super(type(self),self)` class.

    Inhereting class must implement a `get_params` attribute that correlates to class BaseEstimator
    from scikit-learn.

    if another vectorizer is to be provided then it should implement `fit` and `transform` functions.
    Inhereting class must provide a method that calls the fit function that updates the `self.fitted_vectorizer` 
    attribute.

    params:
        - language: language used for stopwords and stemming
        - vectorizer: currently there are two options provided: 'tf_vectorizer', 'tfxidf_vectorizer'

    Notes
    -----
        - All estimators should specify all the parameters that can be set
        at the class level in their ``__init__`` as explicit keyword
        arguments (no ``*args`` or ``**kwargs``).

        - It is mandatory to initialize `super(type(self), self)` class. 
    """
    def __init__(self, language = 'english', vectorizer = 'tf_vectorizer'):
        self.classifier = None
        self.punct_tokenizer = WordPunctTokenizer()
        self.tokenizer = RegexpTokenizer('[a-zA-z]+')
        self.language = language
        self.stopwords = stopwords.words(language)
        self.stemer = SnowballStemmer(language)
        self.tf_idf = TfidfVectorizer(tokenizer = self.preproccesor)
        self.tf = CountVectorizer(tokenizer = self.preproccesor)
        self.vectorizer = vectorizer
        self.fitted_vectorizer = None

    def get_params(self, deep = True):
        return {'language':'english',
                'vectorizer': 'tf_vectorizer'}

    def delete_stopwords(self, document):
        """
        Remove stopwords from a given set of words.
        Stopwords selected will depend of language selection.
        """
        return [word for word in document if word not in self.stopwords]

    def preproccesor(self, document):
        tokens = self.delete_stopwords(self.tokenizer.tokenize(document))
        return [self.stemer.stem(token) for token in tokens]

    def tf_vectorizer(self, X):
        self.tf.fit(X)
        self.fitted_vectorizer = self.tf

    def tfxidf_vectorizer(self, X):
        self.tf_idf.fit(X)
        self.fitted_vectorizer = self.tf_idf

    def transform(self, X):
        return self.fitted_vectorizer.transform(X)

    def cross_validation_score(self, X, y, k = 10, n_jobs = None):
        random.seed(9)
        scores = cross_val_score(self, X, y = y, cv = k, n_jobs = n_jobs, verbose = 1)
        print('Score for each fold: ', scores)
        mean_score = sum(scores)/len(scores)
        print('mean score ', mean_score)
        return scores, mean_score

    def score(self, X, y):
        if self.fitted_vectorizer is None:
            raise NotImplementedError("need to fit first")

        new_X = self.transform(X)
        return self.classifier.score(new_X, y)

    def predict(self, x):
        if self.fitted_vectorizer is None:
            raise NotImplementedError("need to fit first")

        new_x = self.transform(x)
        return self.classifier.predict(new_x)

    def fit(self, X, y):
        if self.classifier is None:
            raise NotImplementedError("need to provide an estimator for BaseAdvertiseClassifier class")

        fit_vectorizer = self.__getattribute__(self.vectorizer)
        fit_vectorizer(X)
        new_X = self.transform(X)
        return self.classifier.fit(new_X, y)

def load_movie_reviews():
    sentences = []
    classes = []
    ids = list(movie_reviews.fileids())
    random.shuffle(ids)

    for fd in ids:
        if fd.startswith('neg/'):
            cls = 'neg'
        else:
            cls = 'pos'

        fp = movie_reviews.open(fd)
        sentences.append(fp.read())
        classes.append(cls)

    print('loaded sentences:', len(sentences))
    return sentences, classes

def movie_reviews_test( estimator, k = 10, n_jobs = -1):
    """
    Programmed test to any estimator that implements BaseAdvertiseClassifier
    """
    X, y = load_movie_reviews()
    estimator.cross_validation_score(X, y, k = k, n_jobs = n_jobs)
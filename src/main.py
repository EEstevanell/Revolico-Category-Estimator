from engine.LinearSVC import LinearSVClassifier
from engine.LogisticRegression import LogisticRegressionClassifier
from engine.MultinomialNB import MultinomialNBClassifier
import pickle
import os

raw_data_path = R"C:\Users\Ernesto Estevanell\Documents\GitHub\Revolico-Category-Estimator\src\data"
corpus_data_path = R"C:\Users\Ernesto Estevanell\Documents\GitHub\Revolico-Category-Estimator\src\data\corpus"

def get_raw_data(path, max_depth = 4):
    """
    Explore directory and subdirectories data files
    """
    i = 0
    X = []
    Y = []
    for (dirpath, _, files) in os.walk(path):
        for file in files:
            fd = open(os.path.join(dirpath, file), 'rb')
            y, x = pickle.load(fd)


            if x == '':
                continue

            X.append(x)
            Y.append(y)

        i+=1
        if i >= max_depth:
            break
    return X, Y

def save_corpus(path, X, y):
    for i in range(len(X)):
        #create directory if doesn't exist
        try:
            os.mkdir(fR"{path}\{y[i]}")
        except:
            pass
        with open(fR"{path}\{y[i]}\{i}", 'w') as fd:
            fd.write(X[i])

    with open(fR"{path}\binary_X", 'wb') as xfd, open(fR"{path}\binary_Y", 'wb') as yfd:
        pickle.dump(X, xfd)
        pickle.dump(y, yfd)


X, y = get_raw_data(raw_data_path, 1)
save_corpus(corpus_data_path, X, y)
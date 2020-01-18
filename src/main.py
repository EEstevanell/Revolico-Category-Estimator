from engine.LinearSVC import LinearSVClassifier
from engine.base import load_movie_reviews
from engine.LogisticRegression import LogisticRegressionClassifier
from engine.MultinomialNB import MultinomialNBClassifier
from engine.CustomEstimator import CustomEstimator
from sklearn.model_selection import learning_curve, cross_validate
from multiprocessing import Process, Manager
import threading
import pickle
import os
import matplotlib.pyplot as plt

raw_data_path = R"C:\Users\Ernesto Estevanell\Downloads\Telegram Desktop\crawlerV2\crawler\craw\websites"
corpus_data_path = R"C:\Users\Ernesto Estevanell\Documents\GitHub\Revolico-Category-Estimator\src\data\corpus"

def get_raw_data(path, max_depth = 4):
    """
    Explore directory and subdirectories looking for data files
    (this data files are the .html files from the spyder)
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
    """
    Save loaded corpus
    """
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

def load_corpus(path):
    """
    load previously saved corpus using pickle
    """
    try:
        with open(fR"{path}\binary_X", 'rb') as xfd, open(fR"{path}\binary_Y", 'rb') as yfd:
            X = pickle.load(xfd)
            y = pickle.load(yfd)
            return X, y
    except:
        #TODO: implement corpus reading from directories
        pass

def run(X, y, k = 30, n_jobs = -1):
    svm_score = LinearSVClassifier().cross_validation_score(X, y, k, n_jobs)
    mnb_score = MultinomialNBClassifier().cross_validation_score(X, y, k, n_jobs)
    lr_score = LogisticRegressionClassifier().cross_validation_score(X, y, k, n_jobs)

    print(f"Obtained mean scores:\n{svm_score[1]} (LinearSVC)\n{mnb_score[1]} (MultinomialNaiveBayes)\n{lr_score[1]} (LogisticRegression)")
    return (svm_score, mnb_score, lr_score)

def _fit(estimator, X, y, return_list):
    estimator.fit(X, y)
    return_list.append(estimator)

def fit(X, y, vectorizer = 'tf_vectorizer', language = 'spanish'):
    """
    returns a tupple of three fitted classifiers

    linearSVC
    MultinomialNB
    LogisticRegression
    """
    manager = Manager()
    return_list = manager.list()
    svm = LinearSVClassifier(language = language, vectorizer=vectorizer)
    mnb = MultinomialNBClassifier(language = language, vectorizer=vectorizer)
    lr = LogisticRegressionClassifier(language = language, vectorizer=vectorizer)
    ce = CustomEstimator(language = language, vectorizer=vectorizer)

    p1 = Process(target = _fit, args = (svm, X, y, return_list))
    p2 = Process(target = _fit, args = (mnb, X, y, return_list))
    p3 = Process(target = _fit, args = (lr, X, y, return_list))
    p4 = Process(target = _fit, args = (ce, X, y, return_list))

    p1.start()
    p2.start()
    p3.start()
    p4.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()
    
    return return_list

def predict(x, estimators):
    return [estimator.predict(x) for estimator in estimators]

def plot_learning_curves(X, y, k = 5):
    mnb = MultinomialNBClassifier(language='spanish')
    svm = LinearSVClassifier(language='spanish')
    lr = LogisticRegressionClassifier(language='spanish')
    ce = CustomEstimator(language='spanish')

    data_sizes = []

    mnb_train_score = []
    mnb_valid_score = []

    svm_train_score = []
    svm_valid_score = []

    lr_train_score = []
    lr_valid_score = []

    ce_train_score = []
    ce_valid_score = []

    for X_new, y_new in get_strattified_data(X, y, step = 500):
        data_sizes.append(len(X_new))
        print("starting evaluation with size:", len(X_new))

        print("running MNB")
        mnb_scores = cross_validate(mnb, X_new, y_new, cv = k, n_jobs = -1, return_train_score = True)
        mnb_train_score.append(sum(mnb_scores['train_score'])/len(mnb_scores['train_score']))
        mnb_valid_score.append(sum(mnb_scores['test_score'])/len(mnb_scores['test_score']))
        mnb_score = sum(mnb_scores['test_score'])/len(mnb_scores['test_score'])

        print("running SVM")
        svm_scores = cross_validate(svm, X_new, y_new, cv = k, n_jobs = -1, return_train_score = True)
        svm_train_score.append(sum(svm_scores['train_score'])/len(svm_scores['train_score']))
        svm_valid_score.append(sum(svm_scores['test_score'])/len(svm_scores['test_score']))
        svm_score = sum(svm_scores['test_score'])/len(svm_scores['test_score'])

        print("running LR")
        lr_scores = cross_validate(lr, X_new, y_new, cv = k, n_jobs = -1, return_train_score = True)
        lr_train_score.append(sum(lr_scores['train_score'])/len(lr_scores['train_score']))
        lr_valid_score.append(sum(lr_scores['test_score'])/len(lr_scores['test_score']))
        lr_score = sum(lr_scores['test_score'])/len(lr_scores['test_score'])

        print("running MLP")
        ce_scores = cross_validate(ce, X_new, y_new, cv = k, n_jobs = -1, return_train_score = True)
        ce_train_score.append(sum(ce_scores['train_score'])/len(ce_scores['train_score']))
        ce_valid_score.append(sum(ce_scores['test_score'])/len(ce_scores['test_score']))
        ce_score = sum(ce_scores['test_score'])/len(ce_scores['test_score'])
        mnb_score = 0
        svm_score = 0
        lr_score = 0
        print(f"ended evaluation with results:\nMNB:{mnb_score}\nSVM:{svm_score}\nLR:{lr_score}\nMLP:{ce_score}")

    plt.figure(1)
    plt.plot(data_sizes, mnb_train_score, 'o-r')
    plt.plot(data_sizes, mnb_valid_score, 'o-b')
    plt.xlabel('Dataset Size')
    plt.ylabel('Score')
    plt.title('Multinomial Naive Bayes')
    plt.legend(['Train Score', 'Validation Score'])

    plt.figure(2)
    plt.plot(data_sizes, svm_train_score, 'o-r')
    plt.plot(data_sizes, svm_valid_score, 'o-b')
    plt.xlabel('Dataset Size')
    plt.ylabel('Score')
    plt.title('Support Vector Machine (Linear Kernel)')
    plt.legend(['Train Score', 'Validation Score'])


    plt.figure(3)
    plt.plot(data_sizes, lr_train_score, 'o-r')
    plt.plot(data_sizes, lr_valid_score, 'o-b')
    plt.xlabel('Dataset Size')
    plt.ylabel('Score')
    plt.title('Logistic Regression')
    plt.legend(['Train Score', 'Validation Score'])
    plt.show()

    plt.figure(1)
    plt.plot(data_sizes, ce_train_score, 'o-r')
    plt.plot(data_sizes, ce_valid_score, 'o-b')
    plt.xlabel('Dataset Size')
    plt.ylabel('Score')
    plt.title('Multilayer Perceptron (27 neurons x 2 layers) (activation = tanh)')
    plt.legend(['Train Score', 'Validation Score'])
    plt.show()

def get_strattified_data(X, y, step = 100):
    """
    Generator to serve increasing amount of strattified data
    """
    #data dict
    data = {}

    #data to return
    ret_X = []
    ret_y = []

    for cls in y:
        data[cls] = []
    
    for i in range(len(X)):
        data[y[i]].append(X[i])

    pc_amount = int(step/len(data.keys()))
    while len(ret_X) < len(X):
        for key in data.keys():
            xs = data[key][:pc_amount]
            ys = [key]*len(xs)

            #erase returned data
            data[key] = data[key][pc_amount:]
            ret_X += xs
            ret_y += ys

        yield (ret_X, ret_y)
 
# X, y = get_raw_data(raw_data_path, 1)
# save_corpus(corpus_data_path, X, y)

if __name__ == '__main__':
    X, y = load_corpus(corpus_data_path)
    plot_learning_curves(X, y)
#     estimators = fit(X, y)
#     results = predict(["laptop de 15 pulgadas"], estimators)
#     print(results)
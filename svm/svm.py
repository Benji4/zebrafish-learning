import numpy as np
import os
import pickle
from sklearn import svm, preprocessing
from sklearn.model_selection import cross_validate, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import time
from tailmetrics import *

def svm_cgamma(X,y):

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-1, 1e-2, 1e-3, 1e-4], 'C': [0.01, 0.1, 1, 10]},
                        {'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10]}]

    scores = ['f1'] #,'accuracy','precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        n_splits = 5
        cv = StratifiedKFold(n_splits=n_splits)

        clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=cv,
                           scoring='%s' % score)
        clf.fit(X, y)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

    best_params = clf.best_params_
    kernel = best_params['kernel']
    svmc = best_params['C']
    if 'gamma' in best_params:
        svmgamma = best_params['gamma']
    else:
        svmgamma = 'auto'

    return kernel, svmc, svmgamma




def calculate_metrics(tailfit, metric_list):
    metrics = np.zeros(len(metric_list))
    for i, metric in enumerate(metric_list):
        metrics[i] = metric(tailfit)
    return metrics


def crossvalidatedSVM(svm_input, svm_input_labels, C=1, gamma=.01, kernel='rbf'):
    n_splits = 5
    if len(svm_input) < 5: # for debugging
        n_splits = len(svm_input)

    """takes in a metric list and generates a crossvalidated SVM"""
    clf=svm.SVC(kernel=kernel, C=C, gamma=gamma)
    cv = StratifiedKFold(n_splits=n_splits)

    scores = cross_validate(clf, svm_input, svm_input_labels, cv=cv,
                                            scoring=('accuracy', 'f1', 'precision', 'recall'))
    accuracies = scores['test_accuracy']
    f1 = scores['test_f1']
    precision = scores['test_precision']
    recall = scores['test_recall']

    output_scores('Validation', accuracies, f1, precision, recall)

    return clf.fit(svm_input,svm_input_labels)

def train_and_output_test_scores(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)  # after cross validation (make no decisions based on the test set scores)
    y_pred = clf.predict(X_test)

    accuracies = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # On the test set, get the y_pred and output them.
    output_scores('Test', accuracies, f1, precision, recall)

def output_scores(type, accuracies, f1, precision, recall):
    print(type, "scores:")
    print("Accuracy : %0.4f (+/- %0.4f)" % (accuracies.mean(), accuracies.std() * 2))
    print("F1 Score : %0.4f (+/- %0.4f)" % (f1.mean(), f1.std() * 2))
    print("Precision: %0.4f (+/- %0.4f)" % (precision.mean(), precision.std() * 2))
    print("Recall   : %0.4f (+/- %0.4f)" % (recall.mean(), recall.std() * 2))

def build_SVM_input(tails, svm_labels, behavior_types=['spon', 'prey'], norm=False):

    # Extracted metrics in Semmelhack:
    # 1. "Maximum tail curvature": maximum over the bout.
    # 2. "Number of peaks in tail angle".
    # 3. "Mean tip angle": absolute value of tip angle in each frame, averaged across the bout.
    # 4. "Maximum tail angle": maximum of the bout.
    # 5. "Mean tip position": average position of last eight points in the tail (horizontal deflection as a frac- tion of the tail length).
    # 6. "Number of frames between peaks": mean (over bout) number of frames between peaks in the tail angle.

    # choose metrics
    metrics = [maxsumangle, numpeaks, meantip, maxangle, tipmean] # list of functions from tailmetrics.py

    svm_input = np.zeros((len(tails), len(metrics)))
    for i, tailfit in enumerate(tails):
        svm_input[i, :] = calculate_metrics(tailfit,metrics)

    metrics_index = dict(zip([i for i in metrics], range(len(metrics))))
    if norm:
        svm_input = preprocessing.normalize(svm_input)
        svm_input = preprocessing.scale(svm_input)
    return svm_input, svm_labels, metrics, metrics_index, behavior_types

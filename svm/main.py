from svm import *
import numpy as np
from sklearn.model_selection import train_test_split
import time
import sys


if __name__ == '__main__':
    sys.stdout = open('out.txt', 'w')

    start = time.time()

    debug = False

    from_npz = True # read from npz or from pickle file

    output_test_scores = True

    behavior_types = ['spon', 'prey']

    print("Loading input...")
    if from_npz: # This works only with numpy version 1.16.2 (not with 1.16.4).
        filename = 'zebrafish_all_tail_clean.npz'
        loaded = np.load(filename)
    else:
        loaded = pickle.load(open("zebrafish_all_tail_aug.p", "rb"))
    tails, y = loaded['tails'], loaded['targets']
    print("Elapsed time:",time.time()-start)

    print("Building SVM input...")
    svm_input, svm_input_labels, metrics_list, metrics_index, behavior_types = build_SVM_input(tails, y, behavior_types)

    print("Elapsed time:",time.time()-start)

    print([(behavior_types[btype], list(svm_input_labels).count(btype)) for btype in set(svm_input_labels)])

    ix=[np.isnan(svm_input[i,:].max()) for i in range(svm_input.shape[0])]
    svm_input = np.delete(svm_input,np.where(ix),0)
    svm_input_labels = np.delete(svm_input_labels,np.where(ix),0)
    
    # Keep all 6 features:
    features = [metrics_index[metric] for metric in metrics_index]

    X = svm_input[:, features]
    y = svm_input_labels

    print("Creating train-test-split...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=462019)  # 85% training and 15% test

    print("Tuning hyperparameters...")
    kernel, svmc, svmgamma = svm_cgamma(X_train,y_train)
    print("Elapsed time:",time.time()-start)

    print("Training...")
    clf = crossvalidatedSVM(X_train, y_train, svmc, svmgamma, kernel)
    print("Elapsed time:",time.time()-start)
    print()

    if output_test_scores:
        train_and_output_test_scores(clf, X_train, X_test, y_train, y_test)
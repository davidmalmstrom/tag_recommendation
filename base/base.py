import sys
sys.path.append("..")

import lib.notebook_helpers as nh
from estimators import BaselineModel, NaiveBayesEstimator, ALSEstimator, SVMEstimator
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import scipy.sparse as sp
import lib.utils as utils
import argparse
from sklearn.metrics import jaccard_score, recall_score
from time import time

def parse_args(sargs):
    parser = argparse.ArgumentParser(description="Run Baseline model.")
    parser.add_argument('--base_model', nargs='?', default='NaiveBayesEstimator',
                        help='Which model to use, \"BaselineModel\", \"NaiveBayesEstimator\", \"ALSEstimator\" or \"SVMEstimator\".')
    parser.add_argument('--num_k_folds', type=int, default=1,
                        help='The number of k-folds to use.')
    parser.add_argument('--svm_C', type=int, default=10,
                        help='The svm C parameter.')
    parser.add_argument('--content_scale_factor', type=float, default=0.06,
                        help='The content scale factor, determining the weight of the content classifier')
    parser.add_argument('--test_dataset', type=int, default=0,
                        help='Specify whether test dataset should be used.')
    parser.add_argument('--topk', type=int, default=10,
                        help='What topk to use when evaluating (recall@K, for example)')
    parser.add_argument('--path', nargs='?', default='../data/',
                        help='Input data path.')
    parser.add_argument('--regularization', type=float, default=0.01,
                        help='Regularization for the ALS estimator.')
    parser.add_argument('--iterations', type=int, default=15,
                        help='Number of iterations for the ALS estimator.')
    parser.add_argument('--show_als_progress', type=int, default=0,
                        help='Whether to print als progress or not.')
    parser.add_argument('--factors', type=int, default=50,
                        help='Number of factors in ALS estimator')
    parser.add_argument('--NB_smoothing', type=float, default=1.0,
                        help='Smoothing parameter for multinomial naive Bayes model.')

    return parser.parse_known_args(sargs)[0]

def get_model(args):
    """Creates models out of parsed args
    """
    model_name = args.base_model

    if model_name == 'BaselineModel':
        model = BaselineModel(factors=args.factors,
                                regularization=args.regularization,
                                iterations=args.iterations,
                                show_progress=args.show_als_progress,
                                n=args.topk,
                                content_scale_factor=args.content_scale_factor)
    elif model_name == "ALSEstimator":
        model = ALSEstimator(factors=args.factors,
                                regularization=args.regularization,
                                iterations=args.iterations,
                                show_progress=args.show_als_progress,
                                n=args.topk)
    elif model_name == "NaiveBayesEstimator":
        model = NaiveBayesEstimator(alpha=args.NB_smoothing, n=args.topk)
    elif model_name == "SVMEstimator":
        model = SVMEstimator(n=args.topk, C=args.svm_C)
    else:
        print("Error: wrong model type")
        sys.exit()
    return model

def main(sargs, log_output=True):
    args = parse_args(sargs)
    if log_output:
        print("Arguments: %s " %(args))

    num_k_folds = args.num_k_folds

    test_dataset = args.test_dataset

    if test_dataset:
        with open(args.path + "test_tag_dataset.pkl", 'rb') as f:
            X, y, mlbx, mlby, val_y, test_y = pickle.load(f)
    else:
        with open(args.path + "dev_tag_dataset.pkl", 'rb') as f:
            X, y, mlbx, mlby = pickle.load(f)

    # y = y[:3000]
    # X = X[:3000]
    X = sp.csr_matrix(X)
    y = y.tocsr()

    num_items = y.shape[0]
    num_usertags = y.shape[1]

    avg_recall = 0
    avg_jaccard = 0

    for fold in range(num_k_folds):
        if log_output:
            print("\n")

        model = get_model(args)

        if num_k_folds > 1:
            start_index = int(num_items * fold / num_k_folds)
            end_index = int(num_items * (fold + 1) / num_k_folds)
        elif args.test_dataset:  # validation from end of user list since first end is already halved (for later testing)
            start_index = num_items - int(num_items / 10)
            end_index = num_items
        else:
            start_index = 0
            end_index = int(num_items/10)

        if args.test_dataset:  # test dataset has pre-calculated, fixed val_x and val_y
            val_x = y[start_index:end_index]
            y_train = y.copy()
        else:
            val_x, val_y = nh.split_user_tags_percentage(y[start_index:end_index], seed=1, todok=True)
            y_train = sp.vstack([y[0:start_index], val_x, y[end_index:]], format="csr")

        t1 = time()
        model.fit(X, y_train)
        t2 = time()
        preds = model.predict(X[start_index:end_index], start_index=start_index)

        val_y_array = val_y.toarray()

        recall = recall_score(val_y_array, preds, average='micro')
        jaccard = jaccard_score(val_y_array, preds, average='micro')

        t3 = time()

        if log_output:
            print("K-fold " + str(fold + 1) + " of " + str(num_k_folds) + ":   " +
                  "Fit: [{0:.2f} s]".format(t2-t1) + ": Recall score: " + str(recall) +
                  ",    Jaccard score: " + str(jaccard) + ",    Eval: [{0:.2f} s]".format(t3-t2))

        avg_recall = avg_recall + ((recall - avg_recall) / (fold + 1))
        avg_jaccard = avg_jaccard + ((jaccard - avg_jaccard) / (fold + 1))

        if num_k_folds > 1 and log_output:
            print("The average performance after k-fold " + str(fold + 1) +
                  " is: Recall = " + str(avg_recall) + ", Jaccard score = " + str(avg_jaccard))

    return avg_recall

if __name__ == '__main__':
    main(sys.argv[1:])
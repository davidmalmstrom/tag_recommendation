import sys
import os
PROJ_ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(PROJ_ROOT_DIR)

import lib.notebook_helpers as nh
import baseline.estimators as estimators
import nncf.run_model_test as nn_test
from nncf.evaluate_recall import evaluate_model_recall

import pickle
import scipy.sparse as sp
import numpy as np
from sklearn.metrics import jaccard_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def compute_metrics(test_y, preds):
    return recall_score(test_y, preds, average='micro'), jaccard_score(test_y, preds, average='micro')

def get_base_results(X, y, test_y, m_space):
    test_part_size = test_y.shape[0]
    
    base_model = estimators.BaselineModel(factors=1000,
                                        regularization=7,
                                        iterations=15,
                                        show_progress=False,
                                        n=3,
                                        content_scale_factor=0.0424,
                                        alpha=2.17)
    base_model.fit(X, y)

    cf_model = estimators.ALSEstimator(factors=1000,
                                        regularization=7,
                                        iterations=15,
                                        show_progress=False)
    cf_model.fit(X, y)

    nb_model = estimators.NaiveBayesEstimator(alpha=2.17)
    nb_model.fit(X, y)

    
    k_list_base = [
        (*compute_metrics(test_y, base_model.predict(X[:test_part_size], start_index=0, n=K)), K, "Baseline")
        for K in m_space
    ]

    k_list_cf = [
        (*compute_metrics(test_y, cf_model.predict(X[:test_part_size], start_index=0, n=K)), K, "Implicit CF")
        for K in m_space
    ]

    k_list_nb = [
        (*compute_metrics(test_y, nb_model.predict(X[:test_part_size], start_index=0, n=K)), K, "Naive Bayes")
        for K in m_space
    ]
    
    return k_list_base + k_list_cf + k_list_nb

def get_nn_results(X, y, test_y, proj_root_dir, config_file_path, m_space, nn_model):
    test_part_size = test_y.shape[0]

    params = nn_test.read_params(config_file_path)

    nn_model = nn_test.build_model(params, y.shape, X.shape[1])
    
    k_list_nn = [
        (*evaluate_model_recall(nn_model, y[:test_part_size], 
                                test_y, K, X[:test_part_size].toarray(), False), K, nn_model)
        for K in m_space
    ]
    
    return k_list_nn

def get_random_results(test_y, m_space, num_user_tags):
    np.random.seed(0)
    def generate_random_preds(shape, m):
        matrix = np.zeros(shape)
        for row in matrix:
            row[np.random.choice(shape[1], m, replace=False)] = 1
        return matrix
    k_list_random = [
        (*compute_metrics(test_y, generate_random_preds(test_y.shape, K)), K, "Random")
        for K in m_space
    ]
    
    return k_list_random

def produce_df_for_plot(m_space, proj_root_dir, dataset_file_name, nn_rel_paths):
    test_set_path = os.path.join(proj_root_dir, "data", dataset_file_name)
    with open(test_set_path, 'rb') as f:
        X, y, mlbx, mlby, val_y, test_y = pickle.load(f)
        y = y.tocsr()
        X = sp.csr_matrix(X)
        test_y = test_y.toarray()
    num_user_tags = y.shape[1]

    full_list = []

    for nn_model, nn_rel_path in zip(['Deep', 'GMF', 'MLP'], nn_rel_paths):
        nn_file_path = os.path.join(proj_root_dir, nn_rel_path)
        full_list += get_nn_results(X, y, test_y, proj_root_dir, nn_file_path, m_space, nn_model)

    base = get_base_results(X, y, test_y, m_space)
    random = get_random_results(test_y, m_space, num_user_tags)

    df = pd.DataFrame(full_list + base + random, columns=["Recall", "Jaccard", "M", "Model"])
    return df

M_SPACE = np.linspace(3, 80, 11, dtype="int32")
sns.set_style("whitegrid")

# Enter runfiles in order: ncf -> gmf -> mlp
configs = [
    {'dataset_file_name': 'test_tag_dataset.pkl',
     'runfiles': ['runzb5.yml', 'runz9.yml', 'runza13.yml'],
     'save_name': 'M_plot_nn_05.svg'
    },
    {'dataset_file_name': 'cold_0.1_test_tag_dataset.pkl',
     'runfiles': ['runzb6.yml', 'runzb4.yml', 'runzb2.yml'],
     'save_name': 'M_plot_nn_01.svg'
    },
    {'dataset_file_name': 'cold_0.0_test_tag_dataset.pkl',
     'runfiles': ['runzb7.yml', 'runzb3.yml', 'runzb1.yml'],
     'save_name': 'M_plot_nn_00.svg'
    }
]

import re
# Split on numbers of runfile names to get folder name
r = re.compile("([a-zA-Z]+)([0-9]+)")

for config in configs:
    dataset_file_name = config['dataset_file_name']
    nn_rel_paths = [
        os.path.join("nncf", "runs", "past_runs", r.match(runfile).group(1), runfile)
        for runfile in config["runfiles"]
    ]
    df = produce_df_for_plot(M_SPACE, PROJ_ROOT_DIR, dataset_file_name, nn_rel_paths)
    plt.figure()
    plot = sns.lineplot(x="M", y="Recall", markers=True, style="Model", dashes=False, hue="Model", data=df)
    plt.savefig(os.path.join(PROJ_ROOT_DIR, "figures", config["save_name"]), bbox_inches='tight')

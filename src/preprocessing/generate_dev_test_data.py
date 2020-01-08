"""
Script to generate development and test set partitions.
Does this for the three different scenarios; p=0, p=0.1 and p=0.5
Uses the result of the preprocessing script.
"""

import sys
import os
sys.path.append(os.path.abspath('..'))

import lib.utils as utils
from sklearn.model_selection import train_test_split
import pickle
import scipy.sparse as sp

path = os.path.join('..', '..', 'data')

dataset = utils.generate_data(n_samples=20000, y_dim=2000, data_dir=path, data_name="preprocessed_user_auto_tags.pkl",
                           amount_x=6, amount_y=6, min_x=6, min_y=6)

# Hold out first 2000 entries for testing.
with open(path + 'dev_tag_dataset.pkl', 'wb') as f:
    X, y, mlbx, mlby = utils.reshape_data(dataset[2000:])
    pickle.dump((X, sp.dok_matrix(y), mlbx, mlby), f)

with open(path + 'test_tag_dataset.pkl', 'wb') as f:
    # first 2000 is test set and last 2000 is val set (for training the models for testing)
    X, y, mlbx, mlby = utils.reshape_data(dataset)
    val_x, val_y = utils.split_user_tags_percentage(y[-2000:])
    y = sp.vstack([y[:-2000], val_x]).todok()
    test_x, test_y = utils.split_user_tags_percentage(y[:2000])
    y = sp.vstack([test_x, y[2000:]]).todok()
    pickle.dump((X, y, mlbx, mlby, val_y, test_y), f)

with open(path + 'cold_0.1_test_tag_dataset.pkl', 'wb') as f:
    # first 2000 is test set and last 2000 is val set (for training the models for testing)
    X, y, mlbx, mlby = utils.reshape_data(dataset)
    val_x, val_y = utils.split_user_tags_percentage(y[-2000:], percentage=0.1)
    y = sp.vstack([y[:-2000], val_x]).todok()
    test_x, test_y = utils.split_user_tags_percentage(y[:2000], percentage=0.1)
    y = sp.vstack([test_x, y[2000:]]).todok()
    pickle.dump((X, y, mlbx, mlby, val_y, test_y), f)

with open(path + 'cold_0.0_test_tag_dataset.pkl', 'wb') as f:
    # first 2000 is test set and last 2000 is val set (for training the models for testing)
    X, y, mlbx, mlby = utils.reshape_data(dataset)
    val_x, val_y = utils.split_user_tags_percentage(y[-2000:], percentage=0)
    y = sp.vstack([y[:-2000], val_x]).todok()
    test_x, test_y = utils.split_user_tags_percentage(y[:2000], percentage=0)
    y = sp.vstack([test_x, y[2000:]]).todok()
    pickle.dump((X, y, mlbx, mlby, val_y, test_y), f)

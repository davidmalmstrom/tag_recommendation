import sys
sys.path.append("..")

import lib.notebook_helpers as nh
from estimators import BaselineModel, NaiveBayesEstimator, ALSEstimator, SVMEstimator
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

test_dataset = False

if test_dataset:
    with open("../data/test_tag_dataset.pkl", 'rb') as f:
        X, y, mlbx, mlby, val_y, _ = pickle.load(f)
else:
    with open("../data/dev_tag_dataset.pkl", 'rb') as f:
        X, y, mlbx, mlby = pickle.load(f)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=1, shuffle=False)
y = y[:8000]
X = X[:8000]
split_index = int(len(y)/10)

val_x, val_y = nh.split_user_tags_percentage(y[:split_index], seed=1, todok=False)
y_go = np.concatenate((val_x, y[split_index:]))

for Model in [BaselineModel, SVMEstimator, NaiveBayesEstimator]:
    model = Model()
    model.fit(X, y_go)
    preds = model.predict(X[:split_index])
    print(model)
    nh.print_all_scores(val_y, preds[:split_index])
print("asdasds")

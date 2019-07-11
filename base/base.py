import sys
sys.path.append("..")

from estimators import BaselineModel, NaiveBayesEstimator, ALSEstimator
from sklearn.model_selection import train_test_split
import pickle

test_dataset = False

if test_dataset:
    with open("../data/test_tag_dataset.pkl", 'rb') as f:
        X, y, mlbx, mlby, val_y, _ = pickle.load(f)
else:
    with open("../data/dev_tag_dataset.pkl", 'rb') as f:
        X, y, mlbx, mlby = pickle.load(f)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1, shuffle=False)

model = BaselineModel()
model.fit(X_test, y_test)
preds = model.predict(X_test)

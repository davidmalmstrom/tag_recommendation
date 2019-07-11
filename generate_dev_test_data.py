import lib.notebook_helpers as nh
from sklearn.model_selection import train_test_split
import pickle
import scipy.sparse as sp

path = "data/"

dataset = nh.generate_data(n_samples=20000, y_dim=2000, data_dir=path, data_name="preprocessed_user_auto_tags_big.pkl",
                           amount_x=6, amount_y=6, min_x=6, min_y=6)

# Hold out first 2000 entries for testing.
with open(path + 'dev_tag_dataset.pkl', 'wb') as f:
    pickle.dump(nh.reshape_data(dataset[2000:]), f)

with open(path + 'test_tag_dataset.pkl', 'wb') as f:
    # first 2000 is test set and last 2000 is val set (for training the models for testing)
    X, y, mlbx, mlby = nh.reshape_data(dataset)
    val_x, val_y = nh.split_user_tags_percentage(y[-2000:])
    y = sp.vstack([y[:-2000], val_x]).todok()
    test_x, test_y = nh.split_user_tags_percentage(y[:2000])
    y = sp.vstack([test_x, y[2000:]]).todok()
    pickle.dump((X, y, mlbx, mlby, val_y, test_y), f)

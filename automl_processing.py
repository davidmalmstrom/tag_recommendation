import pandas as pd
import numpy as np
import scipy.sparse as sp
import pickle

def gen_negative_user_tags(binary_data, mlb, num_negatives=4, extra_data=None):
    """generates <num_negatives> negative user-tag examples
    per positive user-tag in <binary_data>. Transforms the
    negative examples to a list of tuples of user-tag strings
    using <mlb>. <extra_data> are additional tags that should
    not appear as a negative example."""
    np.random.seed(0)
    num_tags = binary_data.shape[1]
    items, u_tags = [],[]

    if extra_data is None:
        extra_data = sp.dok_matrix(binary_data.shape)
    all_known = binary_data + extra_data

    for i, item_row in enumerate(binary_data):
        seen = set()
        for _ in item_row.keys():
            for t in range(num_negatives):
                j = np.random.randint(num_tags)
                while (i, j) in all_known or j in seen:
                    j = np.random.randint(num_tags)
                items.append(i)
                u_tags.append(j)
                seen.add(j)
            
    negatives = sp.dok_matrix(binary_data.shape)
    for i, u in zip(items, u_tags):
        negatives[i, u] = 1

    # not_seen_tag_indices = np.where(np.squeeze(all_true.toarray()) == 0)[0]
    # tag_indices_to_predict = np.random.choice(not_seen_tag_indices, 4, replace=False)
    return mlb.inverse_transform(negatives)

def process_dataset(dataset, mlb, num_negatives=0, binary_data=None, extra_data=None, test_save=False):
    """dataset has to have the three columns <user_tags>,
    <autotags> and <user>.
    <num_negatives> is the amount of negative samples per ground-truth tag.
    <test_save> flag saves the prat of the dataset containing both positive and negative 
    user-tags."""

    dataset['label'] = 1

    if num_negatives > 0:
        negative_train_tags = gen_negative_user_tags(binary_data, mlb, num_negatives, extra_data)
        neg_u_set = pd.DataFrame({"user_tags": pd.Series(negative_train_tags)})

        neg_u_set['user'] = dataset.user

        neg_u_set['autotags'] = dataset.autotags
        neg_u_set['label'] = 0

        dataset = pd.concat([dataset, neg_u_set], axis=0, sort=False, ignore_index=True)
        if test_save:
            with open('data/test_data_balanced.pkl', 'wb') as f:
                # First 2000 rows are positive examples, the 2000 following
                # are negative examples.
                pickle.dump(sp.dok_matrix(mlb.transform(dataset.user_tags)), f)


    # Expand user_tag lists to one user_tag per row
    expand = pd.DataFrame(dataset.user_tags.tolist(), index=dataset.index)\
           .stack().reset_index(level=1, drop=True).rename('user_tag')
    dataset = dataset.join(expand)
    dataset = dataset.drop('user_tags', axis=1)

    return dataset

X, y, mlbx, mlby, val_y, test_y = pd.read_pickle('data/test_tag_dataset.pkl')

train_user_tags = mlby.inverse_transform(y)
val_user_tags = mlby.inverse_transform(val_y)
test_user_tags = mlby.inverse_transform(test_y)

autotags = pd.Series(mlbx.inverse_transform(X))

train_set = pd.DataFrame(data={'autotags': autotags, 'user_tags': pd.Series(train_user_tags)})
train_set['user'] = train_set.index

val_set_slice = slice(18000, 20000)
val_set = pd.DataFrame(data={'autotags': autotags[val_set_slice], 'user_tags': val_user_tags})
val_set['user'] = train_set['user'][val_set_slice]
val_set = val_set.reset_index(drop=True)

test_set_slice = slice(0, 2000)
test_set = pd.DataFrame(data={'autotags': autotags[test_set_slice], 'user_tags': test_user_tags})
test_set['user'] = train_set['user'][test_set_slice]

train_set = process_dataset(train_set, mlby, binary_data=y, num_negatives=4)
val_set = process_dataset(val_set, mlby, binary_data=val_y, num_negatives=4, extra_data=y[val_set_slice])
test_set = process_dataset(test_set, mlby, binary_data=test_y, num_negatives=1, extra_data=y[test_set_slice], test_save=True)

# y[val_set_slice]

train_set['split'] = "TRAIN"
val_set['split'] = "VAL"
test_set['split'] = "TEST"

final_dataset = pd.concat([train_set, val_set, test_set], sort=False, ignore_index=True)
final_dataset.to_csv('data/automl_dataset.csv', index=False)

# for dataset, name in [(train_set, "train_set"), (val_set, "val_set"), (test_set, "test_set")]:
#     dataset.to_csv('data/' + name + ".csv", index=False)
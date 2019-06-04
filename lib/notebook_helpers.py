from sklearn import svm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import preprocessing
import pandas as pd
import numpy as np
import numpy as np
from keras import backend as K
from sklearn.metrics import jaccard_similarity_score, accuracy_score, f1_score, precision_score, recall_score, hamming_loss
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from scipy import sparse
from keras.preprocessing.sequence import pad_sequences

class ModelWrapper:
    """
    Wraps classifier functionality into one class, doing model fitting and prediction in the construction of the object.
    """

    def __init__(self, classifier, X, y, mlbx, mlby, test_size=0.3, top_n=5, has_probs=False, tfidf=False):
        """
        has_probs: Should be entered if the classifier implements the predict_proba method. 
            then the option to choose the amount of hits will be possible when predicting.
        tfidf: Se to True if the features should be tfidf-transformed before training
        """
        self.clf = classifier
        self.mlbx = mlbx
        self.mlby = mlby
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size)

        if tfidf:
            transformer = TfidfTransformer()
            self.x_train = transformer.fit_transform(self.x_train)
            self.x_test = transformer.transform(self.x_test)

        self.clf.fit(self.x_train, self.y_train)
        self.y_pred = self.clf.predict(self.x_test)

        if has_probs:
            self.y_pred_prob = self.clf.predict_proba(self.x_test)
            self.calculate_top_preds(top_n)
        
    def calculate_top_preds(self, top_n=5):
        self.top_n = get_top_n_tags(self.y_pred_prob, n=top_n)
        self.y_top_n_pred = from_keras_format(list(map(lambda x: x + 1, self.top_n)), self.y_train.shape[1])

    def print_scores(self, top_n=False):
        preds = self.y_top_n_pred if top_n else self.y_pred
        print_all_scores(self.y_test, preds)

    def random_check(self, top_n=False):
        preds = self.y_top_n_pred if top_n else self.y_pred
        random_check(self.y_test, preds, self.mlbx, self.mlby)


def is_year_number(number):
    try:
        num = int(number)
        return num < 2030 and num > 1500
    except ValueError:
        return False


def reduce_tags(tag_series, max_num_features, min_occurrences_per_user_tag=5):
    """
    Gets the n most common tags, and removes camera brand name tags
    """
    
    all_user_tags_count = pd.Series((tag for tag_list in tag_series for tag in tag_list)).value_counts()
    
    # Get the n most used tags (not currently used)
    if all_user_tags_count.shape[0] > max_num_features:
        all_user_tags_count = all_user_tags_count[:max_num_features]
    
    # Have at least 5 occurrences per user tag.
    common_user_tags = set(all_user_tags_count[all_user_tags_count > min_occurrences_per_user_tag].index)

    # Remove the camera brands, as well as the year labels, as these would not give any information
    try:
        common_user_tags.remove('canon')
        common_user_tags.remove('nikon')
        common_user_tags.remove('sony')
        common_user_tags.remove('eos')
    except KeyError:
        pass
    common_user_tags = set([tag for tag in common_user_tags if not is_year_number(tag)])
    return tag_series.map(lambda tag_list: 
                          [tag for tag in tag_list if tag in common_user_tags])


def generate_data(n_samples=None, x_dim=1000, y_dim=1000, amount_x=5, amount_y=5, data_dir=None, data_name=None):
    """
    n_samples -- desired number of samples, leave empty if all is wanted.
    x_dim -- number of features (maximum)
    y_dim -- number of classes (maximum)
    amount_y -- the minimum amount of features per sample
    amount_x -- the minimum amount of classes per output
    """
    if n_samples is None:
        n = 1000000000
    else:
        n = n_samples
    if data_dir is None:
        data_dir = "/Users/davidmalmstrom/mnt/proj/tag-rec/src/notebooks/flickr100m/results"
    if data_name is None:
        data_name = "preprocessed_user_auto_tags.pkl"
    dataset = pd.read_pickle(data_dir + "/" + data_name)

    # Drop duplicate user tag sets (to avoid problem of flickr bulk tagging)
    # This is done before tag reduction since a set can be unique before we remove camera brand tags
    dataset = dataset.loc[dataset.User_tags.apply(lambda x: frozenset(x)).drop_duplicates().index]
    
    dataset.autotags = reduce_tags(dataset.autotags, x_dim)
    dataset.User_tags = reduce_tags(dataset.User_tags, y_dim)

    # Filter the tag sets to include at least amount_x and amount_y tags.
    dataset = dataset.query('autotags.str.len() > ' + str(amount_x))
    dataset = dataset.query('User_tags.str.len() > ' + str(amount_y))

    if dataset.shape[0] > n:
        dataset = dataset.head(n)
    elif n_samples is not None:
        print("Warning: desired number of samples could not be provided.")
    
    return dataset


def tag_stats(dataset):
    auto_tag_list = pd.Series((tag for tag_list in dataset.autotags for tag in tag_list))
    user_tag_list = pd.Series((tag for tag_list in dataset.User_tags for tag in tag_list))
    
    print("number of autotags:")
    print("total: " + str(len(auto_tag_list)))
    print("unique: " + str(len(auto_tag_list.unique())))
    print("ratio autotags/item: " + str(len(auto_tag_list) / dataset.shape[0]))
    print("")
    print("number of user tags:")
    print("total: " + str(len(user_tag_list)))
    print("unique: " + str(len(user_tag_list.unique())))
    print("ratio user_tags/item: " + str(len(user_tag_list) / dataset.shape[0]))


def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    http://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/float(len(set_true.union(set_pred)) )
        acc_list.append(tmp_a)
    return np.mean(acc_list)


def print_score(y_test, y_pred):
    print("Hamming loss: {}".format(hamming_loss(y_pred, y_test)))
    print("Hamming score: {}".format(hamming_score(y_pred, y_test)))
    print("---")    


def reshape_data(dataset):
    mlbx = MultiLabelBinarizer()
    one_hot_autotags = pd.DataFrame(mlbx.fit_transform(dataset.autotags), columns=mlbx.classes_, index=dataset.index)
    X = one_hot_autotags.values
    
    mlby = MultiLabelBinarizer()
    y = mlby.fit_transform(dataset.User_tags)
    return X, y, mlbx, mlby


def print_all_scores(y_test, y_pred):
    print("jaccard score: " + str(jaccard_similarity_score(y_test, y_pred)))
    print("accuracy score: " + str(accuracy_score(y_test, y_pred)))
    print("f1-score: " + str(f1_score(y_test, y_pred, average='micro')))
    print("recall-score: " + str(recall_score(y_test, y_pred, average='micro')))
    print("precision-score: " + str(precision_score(y_test, y_pred, average='micro')))

    print_score(y_test, y_pred)


def random_check(y_test, y_pred, mlbx, mlby):
    from random import randint
    index = randint(0, len(y_test) - 1)
    print(mlby.inverse_transform(y_test)[index])
    print(mlby.inverse_transform(y_pred)[index])
    print("index: " + str(index))


def to_keras_format(mat):
    # Keras embedding layer requires one-hot encoded data to be transformed into array of positive integers.
    # + 1 is to avoid using 0, which is used as a padding for the embedding layer
    ret = np.array([(np.nonzero(row)[0] + 1) for row in mat])
    max_length = max([len(row) for row in ret])
    return pad_sequences(ret, maxlen=max_length, padding='post')

def from_keras_format(mat, num_classes):
    # Works on input with or without the padding
    ret = np.zeros((len(mat), num_classes))
    for count, row in enumerate(mat):
        for index in row:
            if index == 0:  # We have reached the padding zeroes
                break
            ret[count][index - 1] = 1
    return ret


def iou(y_true, y_pred, label: int):
    """
    Return the Intersection over Union (IoU) for a given label.
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
        label: the label to return the IoU for
    Returns:
        the IoU for the given label
    """
    # extract the label values using the argmax operator then
    # calculate equality of the predictions and truths to the label
    y_true = K.cast(K.equal(K.argmax(y_true), label), K.floatx())
    y_pred = K.cast(K.equal(K.argmax(y_pred), label), K.floatx())
    # calculate the |intersection| (AND) of the labels
    intersection = K.sum(y_true * y_pred)
    # calculate the |union| (OR) of the labels
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    # avoid divide by zero - if the union is zero, return 1
    # otherwise, return the intersection over union
    return K.switch(K.equal(union, 0), 1.0, intersection / union)

def mean_iou(y_true, y_pred):
    """
    Return the Intersection over Union (IoU) score.
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
    Returns:
        the scalar IoU value (mean over all labels)
    """
    # get number of labels to calculate IoU for
    num_labels = K.int_shape(y_pred)[-1]
    # initialize a variable to store total IoU in
    total_iou = K.variable(0)
    # iterate over labels to calculate IoU for
    for label in range(num_labels):
        total_iou = total_iou + iou(y_true, y_pred, label)
    # divide total IoU by number of labels to get mean IoU
    return total_iou / num_labels


def get_top_n_tags(prediction, n=5):
    return [row.argsort()[-n:][::-1] for row in prediction]


def split_half_tags(full_data):
    x_cf_train = full_data.copy()
    for row_index in range(full_data.shape[0]):
        nonzeros = np.nonzero(full_data[row_index])[0]
        # Set half of the non-zero elements in the row to zero. These are saved in y_cf_train, and will be predicted
        x_cf_train[row_index, np.random.choice(nonzeros, int(len(nonzeros)/2), replace=False)] = 0
    # Set y_cf_train to contain the remainder of the elements.
    y_cf_train = full_data - x_cf_train
    return x_cf_train, y_cf_train

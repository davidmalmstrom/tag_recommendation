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


def reduce_tags(tag_series, max_num_features, min_occurrences_per_user_tag):
    """
    Gets the n most common tags, and removes camera brand name tags
    """
    
    all_tags_count = pd.Series((tag for tag_list in tag_series for tag in tag_list)).value_counts()
    # sort to always keep the same order between different times the function is called on the same dataset
    all_tags_count = all_tags_count.iloc[np.lexsort([all_tags_count.index, all_tags_count.values])].iloc[::-1]
    
    # Have at least 5 occurrences per user tag.
    all_tags_count = all_tags_count[all_tags_count > min_occurrences_per_user_tag]

    # Remove the camera brands, as well as the year labels, as these would not give any information
    banned_words = ['canon', 'nikon', 'sony', 'eos']
    for tag in all_tags_count.index.copy():
        if is_year_number(tag) or tag in banned_words:
            all_tags_count.drop(tag, inplace=True)

    # Get the n most used tags
    if all_tags_count.shape[0] > max_num_features:
        all_tags_count = all_tags_count[:max_num_features]

    return tag_series.map(lambda tag_list: 
                          [tag for tag in tag_list if tag in all_tags_count.index])


def generate_data(n_samples=None, x_dim=1000, y_dim=1000, amount_x=6, amount_y=6, data_dir=None, data_name=None, min_x=5, min_y=5):
    """
    n_samples -- desired number of samples (before data treatment), leave empty if all is wanted.
    x_dim -- number of features (maximum)
    y_dim -- number of classes (maximum)
    amount_y -- the minimum amount of features per sample
    amount_x -- the minimum amount of classes per output
    min_x -- minimum amount of times a tag is used for x (autotags)
    min_y -- minimum amount of times a tag is used for y (user tags)
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
    
    dataset.autotags = reduce_tags(dataset.autotags, x_dim, min_x)
    dataset.User_tags = reduce_tags(dataset.User_tags, y_dim, min_y)

    # Filter the tag sets to include at least amount_x and amount_y tags.
    dataset = dataset.query('autotags.str.len() >= ' + str(amount_x))
    dataset = dataset.query('User_tags.str.len() >= ' + str(amount_y))

    if dataset.shape[0] > n:
        dataset = dataset.head(n)
    elif n_samples is not None:
        print("")
        print("Warning: desired number of samples could not be provided, generated " + str(dataset.shape[0]) + " samples.")
    
    # Shuffle
    dataset = dataset.sample(frac=1)

    dataset['Photo/video_identifier'] = dataset.index

    # Remove index
    dataset = dataset.reset_index(drop=True)

    # Save index - id mapping
    index_id_mapping = dataset['Photo/video_identifier']
    index_id_mapping.to_frame().to_pickle(data_dir + "/index_id_mapping.pkl")

    tag_stats(dataset)
    min_a_tags = pd.Series((tag for tag_list in dataset.autotags for tag in tag_list)).value_counts().values[-1]
    min_u_tags = pd.Series((tag for tag_list in dataset.User_tags for tag in tag_list)).value_counts().values[-1]
    if min_a_tags < min_x:
        print("Warning: min_a < min_x")
    if min_u_tags < min_y:
        print("Warning: min_u < min_y")
    return dataset


def tag_stats(dataset):
    auto_tag_list = pd.Series((tag for tag_list in dataset.autotags for tag in tag_list))
    user_tag_list = pd.Series((tag for tag_list in dataset.User_tags for tag in tag_list))
    
    a_counts = auto_tag_list.value_counts()
    u_counts = user_tag_list.value_counts()

    print("dataset shape: " + str(dataset.shape))
    print("")
    print("number of autotags:")
    print("total: " + str(len(auto_tag_list)))
    print("unique: " + str(len(auto_tag_list.unique())))
    print("ratio autotags/item: " + str(len(auto_tag_list) / dataset.shape[0]))
    print("min number of tags per item: " + str(dataset.autotags.str.len().min()))
    print("min number of times an autotag is used: " + str(a_counts.values[-1]))
    print("")
    print("number of user tags:")
    print("total: " + str(len(user_tag_list)))
    print("unique: " + str(len(user_tag_list.unique())))
    print("ratio user_tags/item: " + str(len(user_tag_list) / dataset.shape[0]))
    print("min number of tags per item: " + str(dataset.User_tags.str.len().min()))
    print("min number of times a user tag is used: " + str(u_counts.values[-1]))



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


def random_check(y_test, y_pred, mlbx, mlby, start_tags = None):
    from random import randint
    index = randint(0, len(y_test) - 1)
    if start_tags is not None:
        print("Start tags: " + str(mlby.inverse_transform(start_tags)[index]))
        print("")
    print("Pred goal: " + str(mlby.inverse_transform(y_test)[index]))
    print("Predicted: " + str(mlby.inverse_transform(y_pred)[index]))
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

def split_user_tags_percentage(cf_data, percentage=0.5):
    """Returns a percentage split of the user tag matrix.
    Outputs the given percentage first, then the remainder.
    The remainder is supposed to be predicted.
    """
    if type(cf_data) is sparse.dok_matrix:
        cf_data = cf_data.toarray()
    y_cf_train = cf_data.copy()
    for row_index in range(cf_data.shape[0]):
        nonzeros = np.nonzero(cf_data[row_index])[0]
        # Set the given percentage of the non-zero elements in the row to zero. 
        y_cf_train[row_index, np.random.choice(
            nonzeros, round(len(nonzeros)*percentage), replace=False)] = 0
    x_cf_train = cf_data - y_cf_train
    return sparse.dok_matrix(x_cf_train), sparse.dok_matrix(y_cf_train)
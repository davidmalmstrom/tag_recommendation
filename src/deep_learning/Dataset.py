import sys
import os
sys.path.append("..")


from builtins import object
import scipy.sparse as sp
import numpy as np
import lib.utils as utils
import pickle

class Dataset(object):
    '''
    classdocs
    '''

    def __init__(self, path, eval_recall, tag_dataset=False, big_tag=False, test_dataset=False,
                 dataset_name_prepend=""):
        '''
        Constructor
        '''
        if tag_dataset:
            # Not in use for now
            if big_tag:
                end_addition = "_big"
            else:
                end_addition = ""

            val_y = None
            if test_dataset:
                with open("../data/" + dataset_name_prepend + "test_tag_dataset.pkl", 'rb') as f:
                    X, y, mlbx, mlby, val_y, _ = pickle.load(f)
            else:
                with open("../data/dev_tag_dataset.pkl", 'rb') as f:
                    X, y, mlbx, mlby = pickle.load(f)

            if not eval_recall:
                # Sample unlabelled as negatives, 99 per item:
                self.testNegatives = [np.random.choice(np.where(row == 0)[0], 99).tolist() for row in y]

                # Sample positives as test positives. The list in list with indices is because NCF is implemented that way.
                # Sets the test positives in the y-matrix to 0, so that these are not in the training process.
                # The order is important here, since y is modified. Therefore, this step should be done after negatives have
                # been sampled, and before the trainMatrix is finalized.
                self.testRatings = [[i, int(np.random.choice(np.nonzero(row)[0]))] for i, row in enumerate(y)]
                for (m, n) in self.testRatings:
                    y[m, n] = 0
            else:
                self.testNegatives = None
                self.testRatings = None
            self.X = X
            self.trainMatrix = sp.dok_matrix(y)
            self.mlbx = mlbx
            self.mlby = mlby
            self.val_y = val_y
        else:
            self.trainMatrix = self.load_rating_file_as_matrix(path + ".train.rating")
            self.testRatings = self.load_rating_file_as_list(path + ".test.rating")
            self.testNegatives = self.load_negative_file(path + ".test.negative")
        if not eval_recall: assert len(self.testRatings) == len(self.testNegatives)

        self.num_users, self.num_items = self.trainMatrix.shape

    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList

    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1: ]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList

    def load_rating_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if (rating > 0):
                    mat[user, item] = 1.0
                line = f.readline()
        return mat

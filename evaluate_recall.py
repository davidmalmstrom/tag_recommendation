"""
Evaluate performance of top-K recommendation

Protocol: half test set
Measures: Recall and jaccard-score
"""

import lib.notebook_helpers as nh
import numpy as np
import heapq
from sklearn.metrics import recall_score, jaccard_score

def evaluate_model_recall(model, val_x, val_y, K, X, fast_eval=False, starting_user_num=0):
    """
    val_x: The part of the training set whose labels have been cut in half. The remaining half is in val_y.
    val_y: Half of the labels, used as a truth in the tests.
    K: K in recall@K
    It is important that the val_x part is put before the rest of the data, when concatenated before training.
    This is because the precision functions rely on that the val_x part user numbers start with 0.
    """
    y_pred = get_preds(model, val_x, K, fast_eval, starting_user_num, X, val_y)
    return recall_score(val_y, y_pred, average='micro'), jaccard_score(val_y, y_pred, average='micro')


def get_preds(model, val_x, K, fast_eval, starting_user_num, X, val_y=None):
    def top_cands(user_row, i, u_num):
        # Just the same user all the way, like in ncf (evaluate.py)

        if fast_eval:
            if val_y is None:
                print("Error, val_y need to be provided in order to use fast evaluation")
                import sys
                sys.exit(1)
            num_unseen_tags = 100
            all_true = user_row + val_y[i]

            # Make predictions only on the items that has not been added to the model 
            # (i.e. 1:s in the final concatenated train matrix)
            not_seen_tag_indices = np.where(np.squeeze(all_true.toarray()) == 0)[0]

            tag_indices_to_predict = np.random.choice(not_seen_tag_indices, num_unseen_tags, replace=False)
            tag_indices_to_predict = np.append(tag_indices_to_predict, np.nonzero(val_y[i])[1])
            np.random.shuffle(tag_indices_to_predict)
        else: 
            tag_indices_to_predict = np.where(np.squeeze(user_row.toarray()) == 0)[0]

        users = np.full(len(tag_indices_to_predict), u_num, dtype='int32')
        features = np.tile(X[u_num], (len(tag_indices_to_predict), 1))
        predictions = model.predict([users, tag_indices_to_predict, features],
                                    batch_size=100, verbose=0)
        
        map_item_score = {}
        for j in range(len(tag_indices_to_predict)):
            item_index = tag_indices_to_predict[j]
            map_item_score[item_index] = predictions[j][0]
    
        # return top rank list
        return np.array(heapq.nlargest(K, map_item_score, key=map_item_score.get))


#[row.argsort()[-n:][::-1] for row in prediction]
#np.array([pre[0] for pre in predictions]).argsort()[-10:][::-1]
#np.array(heapq.nlargest(K, map_item_score, key=map_item_score.get))
    # This is the part of the prediction function that uses the fact that the val_x part was prepended to the rest of the 
    # training data before the training occurred.
    tops = np.array([top_cands(user_row, i, i + starting_user_num) for i, user_row in enumerate(val_x)])

    return nh.from_keras_format(list(map(lambda x: x + 1, tops)), val_x.shape[1])

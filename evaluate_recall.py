"""
Evaluate performance of top-K recommendation

Protocol: half test set
Measures: Recall and jaccard-score
"""

import lib.notebook_helpers as nh
import numpy as np
import heapq
from sklearn.metrics import recall_score, jaccard_score

def evaluate_model_recall(model, val_x, val_y, K):
    """
    val_x: The part of the training set whose labels has been cut in half. The remaining half is in val_y.
    val_y: Half of the labels, used as a truth in the tests.
    K: K in recall@K
    It is important that the val_x part is put before the rest of the data, when concatenated before training.
    This is because the precition functions rely on that the val_x part user numbers start with 0.
    """
    y_pred = get_preds(model, val_x, K)
    return recall_score(val_y, y_pred, average='micro'), jaccard_score(val_y, y_pred, average='micro')


def get_preds(model, val_x, K):
    def top_cands(user_row, u_num):
        # Just the same user all the way, like in ncf (evaluate.py)

        # Make predictions only on the items that has not been added to the model 
        # (i.e. 1:s in the final concatenated train matrix)
        not_seen_items = np.where(np.squeeze(user_row.toarray()) == 0)[0]
        users = np.full(len(not_seen_items), u_num, dtype='int32')
        predictions = model.predict([users, not_seen_items],
                                    batch_size=100, verbose=0)
        
        map_item_score = {}
        for i in range(len(not_seen_items)):
            item = not_seen_items[i]
            map_item_score[item] = predictions[i][0]
    
        # return top rank list
        return np.array(heapq.nlargest(K, map_item_score, key=map_item_score.get))


#[row.argsort()[-n:][::-1] for row in prediction]
#np.array([pre[0] for pre in predictions]).argsort()[-10:][::-1]
#np.array(heapq.nlargest(K, map_item_score, key=map_item_score.get))
    # This is the part of the prediction function that uses the fact that the val_x part was prepended to the rest of the 
    # training data before the training occurred.
    tops = np.array([top_cands(user_row, user_number) for user_number, user_row in enumerate(val_x)])

    return nh.from_keras_format(list(map(lambda x: x + 1, tops)), val_x.shape[1])
'''
Created on Aug 9, 2016
Keras Implementation of Neural Matrix Factorization (NeuMF) recommender model in:
He Xiangnan et al. Neural Collaborative Filtering. In WWW 2017.  

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
from __future__ import print_function
from __future__ import division

from builtins import range
from past.utils import old_div
import numpy as np

import keras
from keras import backend as K
from keras import initializers
from keras.regularizers import l1, l2, l1_l2
from keras.models import Sequential, Model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, Multiply, Reshape, Flatten, Dropout, Concatenate
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from evaluate import evaluate_model
from evaluate_recall import evaluate_model_recall
from Dataset import Dataset
from time import time
import sys
import GMF, MLP
import argparse
import scipy.sparse as sp
import lib.notebook_helpers as nh

#################### Arguments ####################
def parse_args(sargs):
    parser = argparse.ArgumentParser(description="Run NeuMF.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=8,
                        help='Embedding size of MF model.')
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
                        help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_mf', type=float, default=0,
                        help='Regularization for MF embeddings.')                    
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
                        help="Regularization for each MLP layer. reg_layers[0] is the regularization for embeddings.")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    parser.add_argument('--mf_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for MF part. If empty, no pretrain will be used')
    parser.add_argument('--mlp_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for MLP part. If empty, no pretrain will be used')
    parser.add_argument('--is_tag', type=int, default=0,
                        help='Specify if a tag dataset is to be used, or just the regular default datasets')
    parser.add_argument('--eval_recall', type=int, default=0,
                        help='Whether the recall evaluation method should be used or not')
    parser.add_argument('--topk', type=int, default=10,
                        help='What topk to use when evaluating (recall@K, for example)')
    parser.add_argument('--big_tag', type=int, default=0,
                        help='Whether the large amount of tag data should be used or not')
    parser.add_argument('--nn_model', nargs='?', default='NeuMF',
                        help='Which model to use, \"NeuMF\", \"GMF\" or \"MLP\"')
    parser.add_argument('--percentage', type=float, default=0.5,
                        help='The percentage of user_tags that should be used for training. 0 means cold start.')
    parser.add_argument('--num_k_folds', type=int, default=1,
                        help='The number of k-folds to user (only applicable for eval_recall).')
    parser.add_argument('--test_dataset', type=int, default=0,
                        help='Specify whether test dataset should be used.')
    return parser.parse_known_args(sargs)[0]

def init_normal(shape, dtype=None):
    return K.random_normal(shape, dtype=dtype)

def get_model(num_users, num_items, mf_dim=10, layers=[10], reg_layers=[0], reg_mf=0):
    assert len(layers) == len(reg_layers)
    num_layer = len(layers) #Number of layers in the MLP
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')
    
    # Embedding layer
    MF_Embedding_User = Embedding(input_dim = num_users, output_dim = mf_dim, name = 'mf_embedding_user',
                                  embeddings_initializer = init_normal, embeddings_regularizer = l2(reg_mf), input_length=1)
    MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = mf_dim, name = 'mf_embedding_item',
                                  embeddings_initializer = init_normal, embeddings_regularizer = l2(reg_mf), input_length=1)   

    MLP_Embedding_User = Embedding(input_dim = num_users, output_dim = old_div(layers[0],2), name = "mlp_embedding_user",
                                  embeddings_initializer = init_normal, embeddings_regularizer = l2(reg_layers[0]), input_length=1)
    MLP_Embedding_Item = Embedding(input_dim = num_items, output_dim = old_div(layers[0],2), name = 'mlp_embedding_item',
                                  embeddings_initializer = init_normal, embeddings_regularizer = l2(reg_layers[0]), input_length=1)   
    
    # MF part
    mf_user_latent = Flatten()(MF_Embedding_User(user_input))
    mf_item_latent = Flatten()(MF_Embedding_Item(item_input))
    mf_vector = Multiply()([mf_user_latent, mf_item_latent]) # element-wise multiply

    # MLP part 
    mlp_user_latent = Flatten()(MLP_Embedding_User(user_input))
    mlp_item_latent = Flatten()(MLP_Embedding_Item(item_input))
    mlp_vector = Concatenate()([mlp_user_latent, mlp_item_latent])
    for idx in range(1, num_layer):
        layer = Dense(layers[idx], kernel_regularizer= l2(reg_layers[idx]), activation='relu', name="layer%d" %idx)
        mlp_vector = layer(mlp_vector)

    # Concatenate MF and MLP parts
    #mf_vector = Lambda(lambda x: x * alpha)(mf_vector)
    #mlp_vector = Lambda(lambda x : x * (1-alpha))(mlp_vector)
    predict_vector = Concatenate()([mf_vector, mlp_vector])
    
    # Final prediction layer
    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name = "prediction")(predict_vector)
    
    model = Model(inputs=[user_input, item_input], 
                  outputs=prediction)
    
    return model

def load_pretrain_model(model, gmf_model, mlp_model, num_layers):
    # MF embeddings
    gmf_user_embeddings = gmf_model.get_layer('user_embedding').get_weights()
    gmf_item_embeddings = gmf_model.get_layer('item_embedding').get_weights()
    model.get_layer('mf_embedding_user').set_weights(gmf_user_embeddings)
    model.get_layer('mf_embedding_item').set_weights(gmf_item_embeddings)
    
    # MLP embeddings
    mlp_user_embeddings = mlp_model.get_layer('user_embedding').get_weights()
    mlp_item_embeddings = mlp_model.get_layer('item_embedding').get_weights()
    model.get_layer('mlp_embedding_user').set_weights(mlp_user_embeddings)
    model.get_layer('mlp_embedding_item').set_weights(mlp_item_embeddings)
    
    # MLP layers
    for i in range(1, num_layers):
        mlp_layer_weights = mlp_model.get_layer('layer%d' %i).get_weights()
        model.get_layer('layer%d' %i).set_weights(mlp_layer_weights)
        
    # Prediction weights
    gmf_prediction = gmf_model.get_layer('prediction').get_weights()
    mlp_prediction = mlp_model.get_layer('prediction').get_weights()
    new_weights = np.concatenate((gmf_prediction[0], mlp_prediction[0]), axis=0)
    new_b = gmf_prediction[1] + mlp_prediction[1]
    model.get_layer('prediction').set_weights([0.5*new_weights, 0.5*new_b])    
    return model

def get_train_instances(train, num_negatives, num_items):
    user_input, item_input, labels = [],[],[]
    for (u, i) in list(train.keys()):
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train:
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels


def split_half_tags(full_data):
    x_cf_train = full_data.copy()
    for row_index in range(full_data.shape[0]):
        nonzeros = np.nonzero(full_data[row_index])[1]
        # Set half of the non-zero elements in the row to zero. These are saved in y_cf_train, and will be predicted
        x_cf_train[row_index, np.random.choice(
            nonzeros, int(len(nonzeros)/2), replace=False)] = 0
    y_cf_train = full_data - x_cf_train
    return x_cf_train, y_cf_train


def main(sargs):
    args = parse_args(sargs)
    num_epochs = args.epochs
    batch_size = args.batch_size
    mf_dim = args.num_factors
    layers = eval(args.layers)
    reg_mf = args.reg_mf
    reg_layers = eval(args.reg_layers)
    num_negatives = args.num_neg
    learning_rate = args.lr
    learner = args.learner
    verbose = args.verbose
    mf_pretrain = args.mf_pretrain
    mlp_pretrain = args.mlp_pretrain
    model_type = args.nn_model
            
    topK = args.topk
    evaluation_threads = 1#mp.cpu_count()
    print("%s arguments: %s " %(model_type, args))
    model_out_file = 'Pretrain/%s_%s_%d_%s_%d.h5' %(args.dataset, model_type, mf_dim, args.layers, time())
    print("The best NeuMF model will be saved to %s" %(model_out_file))
    print("!#--weights_path: " + model_out_file)

    # Loading data
    t1 = time()
    dataset = Dataset(args.path + args.dataset, args.eval_recall, args.is_tag, args.big_tag, args.test_dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    if args.eval_recall:
        num_test_ratings = "eval_recall"
    else:
        num_test_ratings = str(len(testRatings))
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%s" 
          %(time()-t1, num_users, num_items, train.nnz, num_test_ratings))
    
    # Build model
    if model_type == 'NeuMF':
        model = get_model(num_users, num_items, mf_dim, layers, reg_layers, reg_mf)
    elif model_type == "GMF":
        model = GMF.get_model(num_users,num_items,mf_dim)
    elif model_type == "MLP":
        model = MLP.get_model(num_users,num_items, layers, reg_layers)
    else:
        print("Error: wrong model type")
        sys.exit()
        
    def compile_model():
        if learner.lower() == "adagrad": 
            model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
        elif learner.lower() == "rmsprop":
            model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
        elif learner.lower() == "adam":
            model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
        else:
            model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')

    compile_model()

    # Load pretrain model
    if mf_pretrain != '' and mlp_pretrain != '' and model_type == 'NeuMF':
        gmf_model = GMF.get_model(num_users,num_items,mf_dim)
        gmf_model.load_weights(mf_pretrain)
        mlp_model = MLP.get_model(num_users,num_items, layers, reg_layers)
        mlp_model.load_weights(mlp_pretrain)
        model = load_pretrain_model(model, gmf_model, mlp_model, len(layers))
        print("Load pretrained GMF (%s) and MLP (%s) models done. " %(mf_pretrain, mlp_pretrain))

    old_weights = model.get_weights()
    orig_train = train.copy()

    if not args.eval_recall:
        num_k_folds = 1
    else:
        num_k_folds = args.num_k_folds
        avg_recall = 0
        avg_jaccard = 0
    for fold in range(num_k_folds):
        # Init performance
        if args.eval_recall:
            print("")
            print("Performing k-fold " + str(fold + 1))

            compile_model()
            model.set_weights(old_weights)

            fast_eval = False
            if num_k_folds > 1:
                start_index = int(num_users * fold / num_k_folds)
                end_index = int(num_users * (fold + 1) / num_k_folds)
                fast_eval = True
            elif args.test_dataset:  # validation from end of user list since first end is already halved (for later testing)
                start_index = num_users - int(num_users / 10)
                end_index = num_users
            else:
                start_index = 0
                end_index = int(num_users/10)

            val_x, val_y = nh.split_user_tags_percentage(orig_train[start_index:end_index])
            train = sp.vstack([val_x, orig_train[0:start_index], orig_train[end_index:]]).todok()
            hr, ndcg = evaluate_model_recall(model, val_x, val_y, topK, fast_eval)

            # Test.remove
            # best_hr = 0.1
            # best_ndcg = 0.05
            # if args.eval_recall and num_k_folds > 1:
            #     avg_recall = avg_recall + ((best_hr - avg_recall) / (fold + 1))
            #     avg_jaccard = avg_jaccard + ((best_ndcg - avg_jaccard) / (fold + 1))
            #     # avg_recall = ((avg_recall * fold) + best_hr) / (fold + 1)
            #     # avg_jaccard = ((avg_jaccard * fold) + best_ndcg) / (fold + 1)
            #     print("The average best performance after k-fold " + str(fold + 1) + 
            #         " is: Recall = " + str(avg_recall) + ", Jaccard score = " + str(avg_jaccard))
            # continue
            # Test.remove

            metric1 = "Recall"
            metric2 = "Jaccard score"
        else:
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
            hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()

            metric1 = "HR"
            metric2 = "NDCG"
        
        print('Init: %s = %.4f, %s = %.4f' % (metric1, hr, metric2, ndcg))
        best_hr, best_ndcg, best_iter = hr, ndcg, -1
        if args.out > 0:
            model.save_weights(model_out_file, overwrite=True) 

        # Training model
        for epoch in range(num_epochs):
            t1 = time()
            # Generate training instances
            user_input, item_input, labels = get_train_instances(train, num_negatives, num_items)
            
            # Training
            hist = model.fit([np.array(user_input), np.array(item_input)], #input
                            np.array(labels), # labels 
                            batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
            t2 = time()
            
            # Evaluation
            if epoch %verbose == 0:
                if args.eval_recall:
                    hr, ndcg = evaluate_model_recall(model, val_x, val_y, topK, fast_eval)
                else:
                    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
                    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
                loss = hist.history['loss'][0]
                print('Iteration %d fit: [%.1f s]: %s = %.4f, %s = %.4f, loss = %.4f, eval: [%.1f s]'
                    % (epoch,  t2-t1, metric1, hr, metric2, ndcg, loss, time()-t2))
                if hr > best_hr:
                    best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                    if args.out > 0:
                        model.save_weights(model_out_file, overwrite=True)

        print("End. Best Iteration %d:  %s = %.4f, %s = %.4f. " %(best_iter, metric1, best_hr, metric2, best_ndcg))
        if args.out > 0:
            print("The best NeuMF model has been saved to %s" %(model_out_file))

        if args.eval_recall and num_k_folds > 1:
            avg_recall = avg_recall + ((best_hr - avg_recall) / (fold + 1))
            avg_jaccard = avg_jaccard + ((best_ndcg - avg_jaccard) / (fold + 1))
            print("The average best performance after k-fold " + str(fold + 1) + 
                  " is: Recall = " + str(avg_recall) + ", Jaccard score = " + str(avg_jaccard))

if __name__ == '__main__':
    main(sys.argv[1:])

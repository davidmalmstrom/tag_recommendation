'''
Created on Aug 9, 2016
Keras Implementation of Multi-Layer Perceptron (GMF) recommender model in:
He Xiangnan et al. Neural Collaborative Filtering. In WWW 2017.  

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
from __future__ import print_function
from __future__ import division

from builtins import range
from past.utils import old_div
import numpy as np
import os

import keras
from keras import backend as K
from keras import initializers
from keras.regularizers import l2
from keras.models import Sequential, Model
from keras.layers.core import Dense, Lambda, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.activations import relu
from keras.layers import Embedding, Input, Dense, Multiply, Reshape, Flatten, Dropout, Concatenate
from keras.constraints import maxnorm
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from evaluate import evaluate_model
from Dataset import Dataset
from time import time
import sys
import argparse
import multiprocessing as mp

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run MLP.")
    parser.add_argument('--path', nargs='?', default='data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
                        help="Size of each layer. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
                        help="Regularization for each layer")
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
    return parser.parse_args()

def init_normal(shape, dtype=None):
    return K.random_normal(shape, dtype=dtype)

def get_model(num_users, num_autotags, num_items, mlbx, layers = [20,10], reg_layers=[0,0]):
    assert len(layers) == len(reg_layers)
    num_layer = len(layers) #Number of layers in the MLP
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')
    user_features = Input(shape=(100,), dtype='float32', name='user_features')

    def decode(datum):
        return np.argmax(datum)
    
    

    MLP_Embedding_User = Embedding(input_dim = num_users, output_dim = old_div(layers[0],2), name = 'user_embedding',
                                  embeddings_initializer = init_normal, embeddings_regularizer = l2(reg_layers[0]), input_length=1)
    MLP_Embedding_Item = Embedding(input_dim = num_items, output_dim = old_div(layers[0],2), name = 'item_embedding',
                                  embeddings_initializer = init_normal, embeddings_regularizer = l2(reg_layers[0]), input_length=1)
    
    # EMBEDDING_DIM = 100
    # GLOVE_DIR = "/home/david/Documents/glove"

    # glove_index = {}
    # with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')) as f:
    #     for line in f:
    #         values = line.split()
    #         word = values[0]
    #         coefs = np.asarray(values[1:], dtype='float32')
    #         glove_index[word] = coefs
    
    # # add words and composite words to index
    # embedding_matrix = np.zeros((num_autotags, EMBEDDING_DIM))
    # for i, autotag in enumerate(mlbx.classes_):
    #     if autotag == "blackandwhite":
    #         autotag = "black-and-white"
    #     elif autotag == "branchlet":
    #         autotag = "small branch"
    #     elif autotag == "carthorse":
    #         autotag = "work horse"
    #     elif autotag == "farmer's market":
    #         autotag = "farmers market"
    #     elif autotag == "grainfield":
    #         autotag = "grain field"
    #     elif autotag == "groupshot":
    #         autotag = "group shot"
    #     elif autotag == "jack-o-lantern":
    #         autotag = "jack o'lantern"
    #     elif autotag == "photomicrograph":
    #         autotag = "microscope photo"
    #     elif autotag == "radiogram":
    #         autotag = "radio telegram"
    #     elif autotag == "stemma":
    #         autotag = "coat of arms"
    #     elif autotag == "sunbath":
    #         autotag = "sun bath"
    #     embedding_matrix[i] = sum((glove_index[word] for word in autotag.split(' ')))
                
    # MLP_Embedding_User_Features = Embedding(input_dim=num_autotags,  # Med mask så ska det vara 1 till här tror jag 
    #                                         output_dim=EMBEDDING_DIM,
    #                                         embeddings_initializer=initializers.Constant(embedding_matrix),
    #                                         input_length=77,  # Ändra detta till en variabel, finns på ett ställe till
    #                                         trainable=False)
    
    # Crucial to flatten an embedding vector!
    user_latent = Flatten()(MLP_Embedding_User(user_input))
    item_latent = Flatten()(MLP_Embedding_Item(item_input))
    
    feature_latent = Dense(512, kernel_regularizer=l2(reg_layers[0]), activation='relu', name='dense_feature_layer1',
                           kernel_initializer='lecun_uniform')(user_features)
    # feature_latent = user_features
    # feature_latent = Flatten()(MLP_Embedding_User_Features(user_features))
    #feature_latent = Dense(layers[0], kernel_regularizer=l2(reg_layers[0]), name='dense_feature_layer2')(feature_latent)

    # feature_latent = Dropout(0.5)(feature_latent)
    user_latent = Concatenate()([user_latent, feature_latent])

    # item_latent = Dropout(0.5)(item_latent)
    # The 0-th layer is the concatenation of embedding layers
    vector = Concatenate()([user_latent, item_latent])
    vector = Concatenate()([feature_latent, item_latent])
    
    # MLP layerss
    for idx in range(1, num_layer):
        layer = Dense(layers[idx], kernel_regularizer=l2(reg_layers[idx]), name = 'layer%d' %idx,
                      kernel_initializer='lecun_uniform')(vector) # activation='relu', name = 'layer%d' %idx)
        vector = Activation('relu')(layer)
        #vector = BatchNormalization()(layer)
        
    # Final prediction layer
    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name = 'prediction')(vector)
    
    model = Model(inputs=[user_input, item_input, user_features], 
                  outputs=prediction)
    
    model.name = "MLP"
    print(model.summary())

    return model

def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [],[],[]
    num_users = train.shape[0]
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

if __name__ == '__main__':
    args = parse_args()
    path = args.path
    dataset = args.dataset
    layers = eval(args.layers)
    reg_layers = eval(args.reg_layers)
    num_negatives = args.num_neg
    learner = args.learner
    learning_rate = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    verbose = args.verbose
    
    topK = 10
    evaluation_threads = 1 #mp.cpu_count()
    print("MLP arguments: %s " %(args))
    model_out_file = 'Pretrain/%s_MLP_%s_%d.h5' %(args.dataset, args.layers, time())
    
    # Loading data
    t1 = time()
    dataset = Dataset(args.path + args.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
          %(time()-t1, num_users, num_items, train.nnz, len(testRatings)))
    
    # Build model
    model = get_model(num_users, num_items, layers, reg_layers)
    if learner.lower() == "adagrad": 
        model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
    else:
        model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')    
    
    # Check Init performance
    t1 = time()
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f [%.1f]' %(hr, ndcg, time()-t1))
    
    # Train model
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    for epoch in range(epochs):
        t1 = time()
        # Generate training instances
        user_input, item_input, labels = get_train_instances(train, num_negatives)
    
        # Training        
        hist = model.fit([np.array(user_input), np.array(item_input)], #input
                         np.array(labels), # labels 
                         batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
        t2 = time()

        # Evaluation
        if epoch %verbose == 0:
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' 
                  % (epoch,  t2-t1, hr, ndcg, loss, time()-t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                if args.out > 0:
                    model.save_weights(model_out_file, overwrite=True)

    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %(best_iter, best_hr, best_ndcg))
    if args.out > 0:
        print("The best MLP model is saved to %s" %(model_out_file))

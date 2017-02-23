#  Compatibility imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import argparse
import logging
import os
import sys

import tensorflow as tf
import numpy as np

from six.moves import xrange as range
from data_generator_baidu import DataGenerator

try:
    from tensorflow.python.ops import ctc_ops
except ImportError:
    from tensorflow.contrib.ctc import ctc_ops

from utils import maybe_download as maybe_download
from utils import sparse_tuple_from as sparse_tuple_from
from utils import pad_sequences as pad_sequences

from utils_baidu import calc_feat_dim, spectrogram_from_file, text_to_int_sequence

# Constants
SPACE_TOKEN = '<space>'
COLLON_TOKEN = "'"
SPACE_INDEX = 2            
COLLON_INDEX = 1            # 1 is reserved to collon
FIRST_INDEX = ord('a') - 3  # 2 is reserved to space

# Some configs
num_features = 161
num_examples = 112
# Accounting the 0th indice + collon +  space + blank label = 29 characters
num_classes = ord('z') - ord('a') + 1 + 1 + 1 + 1 + 1

# Hyper-parameters
num_epochs = 50
num_hidden = 100
num_layers = 2
batch_size = 4
mb_size = 4  # minibatch_size, miniBatch and Batch both mean the same :-)
initial_learning_rate = 5e-3
momentum = 0.9

datagen = DataGenerator()
datagen.load_train_data('train_corpus.json')
#datagen.load_validation_data('validation_corpus.json')
datagen.fit_train(100)


# THE MAIN CODE!
graph = tf.Graph()
with graph.as_default():
    # e.g: log filter bank or MFCC features
    # Has size [batch_size, max_stepsize, num_features], but the
    # batch_size and max_stepsize can vary along each step
    inputs = tf.placeholder(tf.float32, [None, None, num_features])

    # Here we use sparse_placeholder that will generate a
    # SparseTensor required by ctc_loss op.
    targets = tf.sparse_placeholder(tf.int32)

    # 1d array of size [batch_size]
    seq_len = tf.placeholder(tf.int32, [None])

    # Defining the cell
    # Can be:
    #   tf.nn.rnn_cell.RNNCell
    #   tf.nn.rnn_cell.GRUCell
    cell = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)

    # Stacking rnn cells
    stack = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers,
                                        state_is_tuple=True)

    # The second output is the last state and we will no use that
    outputs, _ = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32)

    shape = tf.shape(inputs)
    batch_s, max_timesteps = shape[0], shape[1]

    # Reshaping to apply the same weights over the timesteps
    outputs = tf.reshape(outputs, [-1, num_hidden])

    # Truncated normal with mean 0 and stdev=0.1
    # Tip: Try another initialization
    # see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
    W = tf.Variable(tf.truncated_normal([num_hidden,
                                         num_classes],
                                        stddev=0.1))
    # Zero initialization
    # Tip: Is tf.zeros_initializer the same?
    b = tf.Variable(tf.constant(0., shape=[num_classes]))

    # Doing the affine projection
    logits = tf.matmul(outputs, W) + b

    # Reshaping back to the original shape
    logits = tf.reshape(logits, [batch_s, -1, num_classes])

    # Time major
    logits = tf.transpose(logits, (1, 0, 2))

    loss = ctc_ops.ctc_loss(logits, targets, seq_len)
    cost = tf.reduce_mean(loss)

    optimizer = tf.train.MomentumOptimizer(initial_learning_rate,
                                           0.9).minimize(cost)

    # Option 2: tf.contrib.ctc.ctc_beam_search_decoder
    # (it's slower but you'll get better results)
    decoded, log_prob = ctc_ops.ctc_greedy_decoder(logits, seq_len)

    # Inaccuracy: label error rate
    ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
                                          targets))

with tf.Session(graph=graph) as session:
    # Initializate the weights and biases
#    tf.initialize_all_variables().run()
    tf.global_variables_initializer().run()
    
    for curr_epoch in range(num_epochs):
        train_cost = train_ler = 0
        start = time.time()
        
        for i, batch in enumerate(datagen.iterate_train(mb_size, shuffle=False, sort_by_duration=True)):
            
            train_inputs = batch['x']
            train_targets = batch['y']
            train_texts = batch['texts']
            train_seq_len = batch['input_lengths']
            #batch_train_inputs, batch_train_seq_len = pad_sequences(train_inputs)
            
            batch_train_targets = sparse_tuple_from(train_targets)
            print("Epoch {}/{}, batch number {} ".format(curr_epoch+1, num_epochs, i))

            feed = {inputs: train_inputs,
                    targets: batch_train_targets,
                    seq_len: train_seq_len}

            batch_cost, _ = session.run([cost, optimizer], feed)
            train_cost += batch_cost*batch_size
            train_ler += session.run(ler, feed_dict=feed)*batch_size

        train_cost /= num_examples
        train_ler /= num_examples

        val_inputs, val_targets, val_seq_len = train_inputs, batch_train_targets, train_seq_len

        val_feed = {inputs: val_inputs,
                    targets: val_targets,
                    seq_len: val_seq_len}

        #val_cost, val_ler = session.run([cost, ler], feed_dict=val_feed)

        log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, val_cost = {:.3f}, val_ler = {:.3f}, time = {:.3f}"
        print(log.format(curr_epoch+1, num_epochs, train_cost, train_ler,
                         train_cost, train_ler, time.time() - start))
    # Deco ding
    d = session.run(decoded[0], feed_dict=feed)
    str_decoded = ''.join([chr(x) for x in np.asarray(d[1]) + FIRST_INDEX])
    # Replacing blank label to none
    str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
    # Replacing space label to space
    str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')
    # Replacing collon label to collon
    str_decoded = str_decoded.replace(chr(ord('a') - 2), "'")
    
    
    print('Original:\n{}' .format(train_texts) )
    print('Decoded:\n{}' .format(str_decoded) )

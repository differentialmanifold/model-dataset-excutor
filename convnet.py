""" Using convolutional net on MNIST dataset of handwritten digit
(http://yann.lecun.com/exdb/mnist/)
Author: Chip Huyen
Prepared for the class CS 20SI: "TensorFlow for Deep Learning Research"
cs20si.stanford.edu
"""
from __future__ import print_function
from __future__ import division
from __future__ import print_function

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python import debug as tf_debug

import utils
from model import *

N_CLASSES = 10

# Step 1: Read in data
# using TF Learn's built in function to load MNIST data to the folder data/mnist
mnist = input_data.read_data_sets(os.path.expanduser("~/data/mnist"), one_hot=True)

# Step 2: Define paramaters for the model
LEARNING_RATE = 0.001
BATCH_SIZE = 128
SKIP_STEP = 10
DROPOUT = 0.75
N_EPOCHS = 10

t_model = simple_model.Simple([28, 28, 1], N_CLASSES)
logits = t_model.logits

utils.make_dir('checkpoints')
utils.make_dir('checkpoints/{}'.format(t_model.name))


def get_graph_variable(name):
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    v = [item for item in variables if item.name == name][0]
    return v


def reshape_inputs(inputs):
    return inputs.reshape(-1, *t_model.inputs_shape)


with tf.Session() as sess:
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type='curses')

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=1)

    ##### You have to create folders to store checkpoints
    ckpt = tf.train.get_checkpoint_state('checkpoints/{}'.format(t_model.name))
    # if that checkpoint exists, restore from checkpoint
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('restore from checkpoints')

    initial_step = t_model.global_step.eval()

    start_time = time.time()
    n_batches = int(mnist.train.num_examples / BATCH_SIZE)

    total_loss = 0.0
    for index in range(initial_step, n_batches * N_EPOCHS):  # train the model n_epochs times

        X_batch, Y_batch = mnist.train.next_batch(BATCH_SIZE)

        loss_batch = t_model.update(sess, reshape_inputs(X_batch), Y_batch)

        total_loss += loss_batch
        if (index + 1) % SKIP_STEP == 0:
            print('Average loss at step {}: {:5.1f}'.format(index + 1, total_loss / SKIP_STEP))
            total_loss = 0.0
            saver.save(sess, 'checkpoints/{}/mnist-convnet'.format(t_model.name), index)

            validation_X_batch, validation_Y_batch = mnist.test.next_batch(BATCH_SIZE)
            validation_summary = sess.run(t_model.summaries,
                                          feed_dict={t_model.X: reshape_inputs(validation_X_batch),
                                                     t_model.Y: validation_Y_batch,
                                                     t_model.training: False})
            t_model.validation_writer.add_summary(validation_summary, index)

    print("Optimization Finished!")  # should be around 0.35 after 25 epochs
    print("Total time: {0} seconds".format(time.time() - start_time))
    t_model.train_writer.close()
    t_model.validation_writer.close()

    # test the model
    n_batches = int(mnist.test.num_examples / BATCH_SIZE)
    total_correct_preds = 0
    for i in range(n_batches):
        X_batch, Y_batch = mnist.test.next_batch(BATCH_SIZE)

        accuracy = sess.run(t_model.accuracy, feed_dict={t_model.X: reshape_inputs(X_batch), t_model.Y: Y_batch,
                                                         t_model.training: False})
        total_correct_preds += accuracy * BATCH_SIZE

    print("Accuracy {0}".format(total_correct_preds / mnist.test.num_examples))

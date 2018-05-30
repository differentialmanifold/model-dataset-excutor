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

# Step 3: create placeholders for features and labels
# each image in the MNIST data is of shape 28*28 = 784
# therefore, each image is represented with a 1x784 tensor
# We'll be doing dropout for hidden layer so we'll need a placeholder
# for the dropout probability too
# Use None for shape so we can change the batch_size once we've built the graph
with tf.name_scope('data'):
    X = tf.placeholder(tf.float32, [None, 784], name="X_placeholder")
    Y = tf.placeholder(tf.float32, [None, 10], name="Y_placeholder")

# dropout = tf.placeholder(tf.float32, name='dropout')
training = tf.placeholder(tf.bool, name='training')

# Step 4 + 5: create weights + do inference
# the model is conv -> relu -> pool -> conv -> relu -> pool -> fully connected -> softmax

global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

images = tf.reshape(X, shape=[-1, 28, 28, 1])

t_model = simple_model.Simple()
t_model.load(images, 10, train_model=training)
logits = t_model.logits

accuracy = utils.cal_accuracy(logits, Y)

utils.make_dir('checkpoints')
utils.make_dir('checkpoints/{}'.format(t_model.name))

# Step 6: define loss function
# use softmax cross entropy with logits as the loss function
# compute mean cross entropy, softmax is applied internally
with tf.name_scope('loss'):
    entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits)
    loss = tf.reduce_mean(entropy, name='loss')

with tf.name_scope('summaries'):
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    summary_op = tf.summary.merge_all()

# Step 7: define training op
# using gradient descent with learning rate of LEARNING_RATE to minimize cost
# don't forgot to pass in global_step

optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss, global_step=global_step)


def get_graph_variable(name):
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    v = [item for item in variables if item.name == name][0]
    return v


with tf.Session() as sess:
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type='curses')

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=1)
    # to visualize using TensorBoard
    train_writer = tf.summary.FileWriter('./my_graph/{}/train'.format(t_model.name), sess.graph)
    validation_writer = tf.summary.FileWriter('./my_graph/{}/validation'.format(t_model.name), sess.graph)

    ##### You have to create folders to store checkpoints
    ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/{}/checkpoint').format(t_model.name))
    # if that checkpoint exists, restore from checkpoint
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    initial_step = global_step.eval()

    start_time = time.time()
    n_batches = int(mnist.train.num_examples / BATCH_SIZE)

    total_loss = 0.0
    for index in range(initial_step, n_batches * N_EPOCHS):  # train the model n_epochs times

        X_batch, Y_batch = mnist.train.next_batch(BATCH_SIZE)

        _, loss_batch, train_summary = sess.run([optimizer, loss, summary_op],
                                                feed_dict={X: X_batch, Y: Y_batch, training: True})
        train_writer.add_summary(train_summary, index)

        total_loss += loss_batch
        if (index + 1) % SKIP_STEP == 0:
            print('Average loss at step {}: {:5.1f}'.format(index + 1, total_loss / SKIP_STEP))
            total_loss = 0.0
            saver.save(sess, 'checkpoints/convnet_mnist/mnist-convnet', index)

            validation_X_batch, validation_Y_batch = mnist.test.next_batch(BATCH_SIZE)
            validation_summary = sess.run(summary_op,
                                          feed_dict={X: validation_X_batch, Y: validation_Y_batch, training: False})
            validation_writer.add_summary(validation_summary, index)

    print("Optimization Finished!")  # should be around 0.35 after 25 epochs
    print("Total time: {0} seconds".format(time.time() - start_time))
    train_writer.close()
    validation_writer.close()

    # test the model
    n_batches = int(mnist.test.num_examples / BATCH_SIZE)
    total_correct_preds = 0
    for i in range(n_batches):
        X_batch, Y_batch = mnist.test.next_batch(BATCH_SIZE)
        _, loss_batch, logits_batch = sess.run([optimizer, loss, logits],
                                               feed_dict={X: X_batch, Y: Y_batch, training: False})
        preds = tf.nn.softmax(logits_batch)
        correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))
        accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
        total_correct_preds += sess.run(accuracy)

    print("Accuracy {0}".format(total_correct_preds / mnist.test.num_examples))

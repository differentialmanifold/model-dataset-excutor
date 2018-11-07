import numpy as np
import matplotlib as mpl

mpl.use('TkAgg')

import matplotlib.pyplot as plt
import math
import time

import tensorflow as tf
from model import layer_utils

from model.base_model import BaseModel


class Ext_Simple(BaseModel):
    def __init__(self, inputs_shape, n_class, dropout=0.75, learning_rate=0.001,
                 loss_name='mean_squared_error'):
        self.name = 'ext_simple'
        super().__init__(inputs_shape, n_class, dropout, learning_rate, loss_name)

    def build(self):
        self.fc1 = layer_utils.dense_layer(self.X, units=256, name='fc1', activation=tf.nn.relu)

        self.fc2 = layer_utils.dense_layer(self.fc1, units=256, name='fc2', activation=tf.nn.relu)

        self.logits = layer_utils.dense_layer(self.fc2, units=self.n_class, name='softmax_linear')


def fitting_function(x):
    return 0.2 + 0.4 * (x ** 2) + 0.3 * x * math.sin(15 * x) + 0.05 * math.cos(50 * x)


def generate_data(N):
    x = np.random.rand(N, 1)
    y = np.vectorize(fitting_function)(x)
    return x, y


def custom_plot(x=None, y=None):
    if x is None:
        x = np.arange(0, 1, 0.01)
    if y is None:
        y = np.vectorize(fitting_function)(x)
    plt.plot(x, y)
    plt.show()


N_CLASSES = 1

LEARNING_RATE = 0.001
BATCH_SIZE = 128
SKIP_STEP = 10
DROPOUT = 0.75
N_EPOCHS = 10

t_model = Ext_Simple([1], N_CLASSES, DROPOUT, LEARNING_RATE)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    initial_step = t_model.global_step.eval()

    start_time = time.time()
    n_batches = 300

    total_loss = 0.0
    for index in range(initial_step, n_batches * N_EPOCHS):  # train the model n_epochs times

        X_batch, Y_batch = generate_data(BATCH_SIZE)

        loss_batch = t_model.update(sess, X_batch, Y_batch)

        total_loss += loss_batch
        if (index + 1) % SKIP_STEP == 0:
            print('Average loss at step {}: {:5.1f}'.format(index + 1, total_loss / SKIP_STEP))
            total_loss = 0.0

            validation_X_batch, validation_Y_batch = generate_data(BATCH_SIZE)
            validation_summary = sess.run(t_model.summaries,
                                          feed_dict={t_model.X: validation_X_batch,
                                                     t_model.Y: validation_Y_batch,
                                                     t_model.training: False})
            t_model.validation_writer.add_summary(validation_summary, index)

    print("Optimization Finished!")  # should be around 0.35 after 25 epochs
    print("Total time: {0} seconds".format(time.time() - start_time))
    t_model.train_writer.close()
    t_model.validation_writer.close()

    # test result
    test_n = 100
    x = np.arange(0, 1, 1 / test_n)
    y_real = np.vectorize(fitting_function)(x)
    y_pred = t_model.predict(sess, np.reshape(x, (test_n, 1))).flatten()

    plt.plot(x, y_real, 'r', x, y_pred, 'b')
    plt.show()

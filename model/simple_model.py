import tensorflow as tf
import os

from model import layer_utils


class Simple:
    def __init__(self, inputs_shape, n_class, dropout=0.75, learning_rate=0.001):
        self.name = 'simple'
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.inputs_shape = inputs_shape
        self.n_class = n_class

        parent_dir = os.path.dirname(os.path.dirname(__file__))
        self.train_writer = tf.summary.FileWriter(parent_dir + '/my_graph/{}/train'.format(self.name))
        self.validation_writer = tf.summary.FileWriter(parent_dir + '/my_graph/{}/validation'.format(self.name))

        self.load()

    def load(self):
        with tf.name_scope('data'):
            self.X = tf.placeholder(tf.float32, [None] + self.inputs_shape, name="X_placeholder")
            self.Y = tf.placeholder(tf.float32, [None, self.n_class], name="Y_placeholder")

        self.training = tf.placeholder(tf.bool, name='training')

        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        self.conv1 = layer_utils.conv_layer(self.X, out_channels=32, kernel_size=[5, 5], name='conv1')

        self.pool1 = layer_utils.max_pool(self.conv1, pool_size=[2, 2], stride=2, name='pool1')

        self.conv2 = layer_utils.conv_layer(self.pool1, out_channels=64, kernel_size=[5, 5], name='conv2')

        self.pool2 = layer_utils.max_pool(self.conv2, pool_size=[2, 2], stride=2, name='pool2')

        self.fc = layer_utils.dense_layer(self.pool2, units=1024, name='fc', activation=tf.nn.relu)

        self.drop = layer_utils.dropout_layer(self.fc, keep_prob=self.dropout, training=self.training, name='dropout')

        self.logits = layer_utils.dense_layer(self.drop, units=self.n_class, name='softmax_linear')

        with tf.name_scope('loss'):
            entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=self.logits)
            self.loss = tf.reduce_mean(entropy, name='loss')

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

        self.accuracy = layer_utils.cal_accuracy(self.logits, self.Y)

        self.summaries = tf.summary.merge([
            tf.summary.scalar('loss', self.loss),
            tf.summary.scalar('accuracy', self.accuracy)
        ])

    def predict(self, sess, s):
        return sess.run(self.logits, {self.X: s, self.training: False})

    def update(self, sess, s, y):
        feed_dict = {self.X: s, self.Y: y, self.training: True}
        summaries, global_step, _, loss = sess.run(
            [self.summaries, tf.train.get_global_step(), self.optimizer, self.loss],
            feed_dict)
        if self.train_writer:
            self.train_writer.add_summary(summaries, global_step)
        return loss

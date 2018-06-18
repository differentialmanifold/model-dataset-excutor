import tensorflow as tf

from model import layer_utils


class Vgg16:
    def __init__(self, inputs_shape, n_class, dropout=0.75, learning_rate=0.001):
        self.name = 'vgg16'
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.inputs_shape = inputs_shape
        self.n_class = n_class
        self.load()

    def load(self):
        with tf.name_scope('data'):
            self.X = tf.placeholder(tf.float32, [None] + self.inputs_shape, name="X_placeholder")
            self.Y = tf.placeholder(tf.float32, [None, self.n_class], name="Y_placeholder")

        self.training = tf.placeholder(tf.bool, name='training')

        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        self.conv1_1 = layer_utils.conv_layer(self.X, out_channels=64, kernel_size=[3, 3], name='conv1_1')
        self.conv1_2 = layer_utils.conv_layer(self.conv1_1, out_channels=64, kernel_size=[3, 3], name='conv1_2')
        self.pool1 = layer_utils.max_pool(self.conv1_2, [2, 2], 2, name='pool1')

        self.conv2_1 = layer_utils.conv_layer(self.pool1, out_channels=128, kernel_size=[3, 3], name='conv2_1')
        self.conv2_2 = layer_utils.conv_layer(self.conv2_1, out_channels=128, kernel_size=[3, 3], name='conv2_2')
        self.pool2 = layer_utils.max_pool(self.conv2_2, [2, 2], 2, name='pool2')

        self.conv3_1 = layer_utils.conv_layer(self.pool2, out_channels=256, kernel_size=[3, 3], name='conv3_1')
        self.conv3_2 = layer_utils.conv_layer(self.conv3_1, out_channels=256, kernel_size=[3, 3], name='conv3_2')
        self.conv3_3 = layer_utils.conv_layer(self.conv3_2, out_channels=256, kernel_size=[3, 3], name='conv3_3')
        self.pool3 = layer_utils.max_pool(self.conv3_3, [2, 2], 2, name='pool3')

        self.conv4_1 = layer_utils.conv_layer(self.pool3, out_channels=512, kernel_size=[3, 3], name='conv4_1')
        self.conv4_2 = layer_utils.conv_layer(self.conv4_1, out_channels=512, kernel_size=[3, 3], name='conv4_2')
        self.conv4_3 = layer_utils.conv_layer(self.conv4_2, out_channels=512, kernel_size=[3, 3], name='conv4_3')
        self.pool4 = layer_utils.max_pool(self.conv4_3, [2, 2], 2, name='pool4')

        self.conv5_1 = layer_utils.conv_layer(self.pool4, out_channels=512, kernel_size=[3, 3], name='conv5_1')
        self.conv5_2 = layer_utils.conv_layer(self.conv5_1, out_channels=512, kernel_size=[3, 3], name='conv5_2')
        self.conv5_3 = layer_utils.conv_layer(self.conv5_2, out_channels=512, kernel_size=[3, 3], name='conv5_3')
        self.pool5 = layer_utils.max_pool(self.conv5_3, [2, 2], 2, name='pool5')

        self.fc6 = layer_utils.dense_layer(self.pool5, 4096, activation=tf.nn.relu, name='fc6')
        self.drop6 = layer_utils.dropout_layer(self.fc6, keep_prob=self.dropout, training=self.training, name='drop6')

        self.fc7 = layer_utils.dense_layer(self.drop6, 4096, activation=tf.nn.relu, name='fc7')
        self.drop7 = layer_utils.dropout_layer(self.fc7, keep_prob=self.dropout, training=self.training, name='drop7')

        self.logits = layer_utils.dense_layer(self.drop7, self.n_class, name='softmax_linear')

        with tf.name_scope('loss'):
            entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=self.logits)
            self.loss = tf.reduce_mean(entropy, name='loss')

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

        self.accuracy = layer_utils.cal_accuracy(self.logits, self.Y)

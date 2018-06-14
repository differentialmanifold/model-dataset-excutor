import tensorflow as tf

from model import layer_utils


class Simple:
    def __init__(self, dropout=0.75):
        self.name = 'simple'
        self.dropout = dropout

    def load(self, inputs, n_class, train_model=None):
        self.conv1 = layer_utils.conv_layer(inputs, out_channels=32, kernel_size=[5, 5], name='conv1')

        self.pool1 = layer_utils.max_pool(self.conv1, pool_size=[2, 2], stride=2, name='pool1')

        self.conv2 = layer_utils.conv_layer(self.pool1, out_channels=64, kernel_size=[5, 5], name='conv2')

        self.pool2 = layer_utils.max_pool(self.conv2, pool_size=[2, 2], stride=2, name='pool2')

        self.fc = layer_utils.dense_layer(self.pool2, units=1024, name='fc', activation=tf.nn.relu)

        self.drop = layer_utils.dropout_layer(self.fc, keep_prob=self.dropout, training=train_model, name='dropout')

        self.logits = layer_utils.dense_layer(self.drop, units=n_class, name='softmax_linear')

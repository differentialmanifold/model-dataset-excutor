import tensorflow as tf
from model import layer_utils

from model.base_model import BaseModel


class Simple(BaseModel):
    def __init__(self, inputs_shape, n_class, dropout=0.75, learning_rate=0.001,
                 loss_name='mean_softmax_cross_entropy'):
        self.name = 'simple'
        super().__init__(inputs_shape, n_class, dropout, learning_rate, loss_name)

    def build(self):
        self.conv1 = layer_utils.conv_layer(self.X, out_channels=32, kernel_size=[5, 5], name='conv1')

        self.pool1 = layer_utils.max_pool(self.conv1, pool_size=[2, 2], stride=2, name='pool1')

        self.conv2 = layer_utils.conv_layer(self.pool1, out_channels=64, kernel_size=[5, 5], name='conv2')

        self.pool2 = layer_utils.max_pool(self.conv2, pool_size=[2, 2], stride=2, name='pool2')

        self.fc = layer_utils.dense_layer(self.pool2, units=1024, name='fc', activation=tf.nn.relu)

        self.drop = layer_utils.dropout_layer(self.fc, keep_prob=self.dropout, training=self.training, name='dropout')

        self.logits = layer_utils.dense_layer(self.drop, units=self.n_class, name='softmax_linear')

import tensorflow as tf
from layers.base_layer import BaseLayer
from layers.conv_gate_layer import ConvLayer
from layers.fc_gate_layer import FullConnectedLayer
from layers.bn_layer import BatchNormalizeLayer


class StageLayer(BaseLayer):
    def __init__(self, input, weight_dict, n_block, stride_init, increase, is_training):
        super(StageLayer, self).__init__()

        self.blocks = list()

        self.create(input, weight_dict, n_block, stride_init, increase, is_training)

    def _block(self, x, weight_dict, stride, increase, is_training):
        identity = x

        with tf.variable_scope('c1'):
            c1_layer = ConvLayer(x, weight_dict, is_training, stride=stride)

        b1_layer = BatchNormalizeLayer(c1_layer.layer_output, 'bn1', weight_dict)
        r1 = tf.nn.relu(b1_layer.layer_output)

        with tf.variable_scope('c2'):
            c2_layer = ConvLayer(r1, weight_dict, is_training, stride=1)

        b2_layer = BatchNormalizeLayer(c2_layer.layer_output, 'bn2', weight_dict)

        if increase:
            identity = self._downsample(x, weight_dict, is_training)

        block_output = b2_layer.layer_output + identity
        block_output = tf.nn.relu(block_output)

        layers = [c1_layer, c2_layer]

        return block_output, layers

    def _downsample(self, x, weight_dict, is_training):
        with tf.variable_scope('ds'):
            ds_layer = ConvLayer(x, weight_dict, is_training, 0, stride=2)
        self.weight_tensors += ds_layer.weight_tensors
        return ds_layer.layer_output

    def create(self, x, weight_dict, n_block, stride_init, increase, is_training):
        with tf.variable_scope('b1'):
            x, block_layers = self._block(x, weight_dict, stride_init, increase, is_training)
            self.blocks.append(block_layers)

        for i in range(2, n_block + 1):
            with tf.variable_scope('b%d' % i):
                x, block_layers = self._block(x, weight_dict, stride=1, increase=False, is_training=is_training)
                self.blocks.append(block_layers)

        self.layer_output = x

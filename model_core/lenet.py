import tensorflow as tf
from tensorflow.contrib import slim


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv_block(inputs, output_channels, training, name, kernel_size=3,
               strides=1, layer=tf.layers.conv2d, activation=tf.nn.relu):
    with tf.name_scope(name):
        inputs = layer(inputs, output_channels, kernel_size=kernel_size,
                       strides=strides, padding='same',
                       data_format='channels_first')

        inputs = slim.batch_norm(inputs, decay=0.9, scale=True,
                                 is_training=training,
                                 data_format='NCHW', fused=True)

        inputs = activation(inputs)
    return inputs


def pooling(inputs, name, pool_size=(2, 2), strides=(2, 2), layer=tf.layers.max_pooling2d):
    with tf.name_scope(name):
        inputs = layer(inputs, pool_size, strides, padding='same', data_format='channels_first')
    return inputs


def fc(inputs, num_outputs, name, activation=tf.nn.relu):
    with tf.name_scope(f'{name}/weights'):
        W = weight_variable((int(inputs.shape[1]), num_outputs))
    with tf.name_scope(f'{name}/bias'):
        b = bias_variable([num_outputs])
    return tf.nn.xw_plus_b(inputs, W, b, name=name)


def build_model(inputs, classes, name, training):
    with tf.name_scope(name):
        inputs = conv_block(inputs, 6, training, 'layer1', kernel_size=5)
        inputs = pooling(inputs, 'pooling1')

        inputs = conv_block(inputs, 16, training, 'layer2', kernel_size=5)
        inputs = pooling(inputs, 'pooling2')
        inputs = tf.layers.flatten(inputs, 'flatten')
        inputs = fc(inputs, 120, 'fc1')
        inputs = fc(inputs, 84, 'fc2')
        inputs = fc(inputs, classes, 'fc3', activation=tf.identity)

        return inputs


class LeNet():
    def __init__(self, image_size, n_chans_img, classes):
        self.image_size = image_size
        self.n_chans_img = n_chans_img
        self.classes = classes

    def build(self, training_ph):
        x_ph = tf.placeholder(
            tf.float32, (None, self.n_chans_img, self.image_size, self.image_size), name='input'
        )

        logits = build_model(x_ph, self.classes, 'lenet_2d', training_ph)
        return [x_ph], logits
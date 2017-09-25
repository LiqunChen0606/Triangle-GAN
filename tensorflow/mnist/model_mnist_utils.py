import tensorflow as tf
from tensorflow.contrib import layers
ds = tf.contrib.distributions
st = tf.contrib.bayesflow.stochastic_tensor
import pdb
def standard_normal(shape, **kwargs):
    """Create a standard Normal StochasticTensor."""
    return tf.cast(st.StochasticTensor(
        ds.MultivariateNormalDiag(mu=tf.zeros(shape), diag_stdev=tf.ones(shape), **kwargs)),  tf.float32)


def encoder(input_tensor, output_size):
    '''Create encoder network.
    Args:
        input_tensor: a batch of flattened images [batch_size, 28*28]
    Returns:
        A tensor that expresses the encoder network
    '''
    net = tf.reshape(input_tensor, [-1, 28, 28, 1])
    net = layers.conv2d(net, 32, 5, stride=2, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
    net = layers.conv2d(net, 64, 5, stride=2, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
    net = layers.conv2d(net, 128, 5, stride=2, padding='VALID', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
    net = layers.dropout(net, keep_prob=0.9)
    net = layers.flatten(net)
    return layers.fully_connected(net, output_size, activation_fn=tf.nn.sigmoid, weights_initializer=tf.contrib.layers.xavier_initializer())


def decoder(input_tensor):
    '''Create decoder network.
        If input tensor is provided then decodes it, otherwise samples from
        a sampled vector.
    Args:
        input_tensor: a batch of vectors to decode
    Returns:
        A tensor that expresses the decoder network
    '''
    net = tf.expand_dims(input_tensor, 1)
    net = tf.expand_dims(net, 1)
    net = layers.conv2d_transpose(net, 128, 3, padding='VALID', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
    net = layers.conv2d_transpose(net, 64, 5, padding='VALID', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
    net = layers.conv2d_transpose(net, 32, 5, stride=2, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
    net = layers.conv2d_transpose(
        net, 1, 5, stride=2, activation_fn=tf.nn.sigmoid, weights_initializer=tf.contrib.layers.xavier_initializer())
    net = layers.flatten(net)
    return net


def discriminator(x, y):
    '''Create encoder network.
    Args:
        input_tensor: a batch of flattened images [batch_size, 28*28]
    Returns:
        A tensor that expresses the encoder network
    '''
    net1 = tf.reshape(x, [-1, 28, 28, 1])
    net2 = tf.reshape(y, [-1, 28, 28, 1])
    net = tf.concat([net1, net2], axis = 1)
    # pdb.set_trace()
    net = layers.conv2d(net, 32, 5, stride=2, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
    net = layers.conv2d(net, 64, 5, stride=2, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
    net = layers.conv2d(net, 128, 5, stride=2, padding='VALID', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
    net = layers.dropout(net, keep_prob=0.9)
    net = layers.flatten(net)
    return layers.fully_connected(net, 1, activation_fn=tf.nn.sigmoid, weights_initializer=tf.contrib.layers.xavier_initializer())


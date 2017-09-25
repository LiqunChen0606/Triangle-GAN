import tensorflow as tf
from tensorflow.contrib import layers
import pdb
import numpy as np

def conv_cond_concat(x, y):
  """Concatenate conditioning vector on feature map axis."""
  x_shapes = x.get_shape()
  y_shapes = y.get_shape()
  return tf.concat([
    x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

initializer = tf.truncated_normal_initializer(stddev=0.02)
# initializer = tf.contrib.layers.xavier_initializer()
def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def discriminator(x, y):

    yb = tf.reshape(y, [-1, 1, 1, 40])
    h = tf.reshape(x, [-1, 64, 64, 3])

    h = conv_cond_concat(h, yb)
    h = layers.conv2d(h, 64, 5, stride=2, padding='SAME', activation_fn=None, weights_initializer=initializer)
    h = layers.batch_norm(h, activation_fn=tf.nn.relu)

    # h = conv_cond_concat(h, yb)
    h = layers.conv2d(h, 64*2, 5, stride=2, padding='SAME', activation_fn=None, weights_initializer=initializer)
    h = layers.batch_norm(h, activation_fn=tf.nn.relu)

    # h = conv_cond_concat(h, yb)
    h = layers.conv2d(h, 64*4, 5, stride=2, padding='SAME', activation_fn=None, weights_initializer=initializer)
    h = layers.batch_norm(h, activation_fn=tf.nn.relu)

    # h = conv_cond_concat(h, yb)
    h = layers.conv2d(h, 64*8, 5, stride=2, padding='SAME', activation_fn=None, weights_initializer=initializer)
    h = layers.batch_norm(h, activation_fn=tf.nn.relu)
    h = layers.flatten(h)
    return layers.fully_connected(h, 1, activation_fn=tf.sigmoid)


def encoder1(tensor):

    conv1 =  layers.conv2d(tensor, 32, 4, stride=2, activation_fn=None, weights_initializer=initializer)
    conv1 = layers.batch_norm(conv1, activation_fn=lrelu)

    conv2 =  layers.conv2d(conv1, 64, 4, stride=2, activation_fn=None, normalizer_fn= layers.batch_norm,weights_initializer=initializer)
    conv2 = layers.batch_norm(conv2, activation_fn=lrelu)

    conv3 =  layers.conv2d(conv2, 128, 4, stride=2, activation_fn=None, normalizer_fn= layers.batch_norm, weights_initializer=initializer)                                  # 8 x 8 x 128
    conv3 = layers.batch_norm(conv3, activation_fn=lrelu)

    conv4 =  layers.conv2d(conv3, 256, 4, stride=2, activation_fn=None, normalizer_fn= layers.batch_norm, weights_initializer=initializer)                                  # 4 x 4 x 256
    conv4 = layers.batch_norm(conv4, activation_fn=lrelu)

    conv5 =  layers.conv2d(conv4, 512, 4, stride=2, activation_fn=None, normalizer_fn= layers.batch_norm, weights_initializer=initializer)                                  # 2 x 2 x 512
    conv5 = layers.batch_norm(conv5, activation_fn=lrelu)

    fc1 = tf.reshape(conv5, shape=[-1, 2*2*512])
    fc1 =  layers.fully_connected(inputs=fc1, num_outputs=512,activation_fn=None, weights_initializer=initializer)
    fc1 = layers.batch_norm(fc1, activation_fn=lrelu)

    fc2 =  layers.fully_connected(inputs=fc1, num_outputs=40, activation_fn=tf.nn.sigmoid, weights_initializer=initializer)

    return fc2

def encoder2(y, z):
    yb = tf.reshape(y, [-1, 1, 1, 40])
    h = tf.concat([y, z], axis = 1)
    h = layers.fully_connected(h, 1024, weights_initializer=initializer)
    h = layers.batch_norm(h, activation_fn=lrelu)

    h = layers.fully_connected(h, 64*8*4*4, activation_fn=None, weights_initializer=initializer)
    h = tf.reshape(h, [-1, 4, 4, 64*8])
    h = layers.batch_norm(h, activation_fn=lrelu)
    h = conv_cond_concat(h, yb)

    h = layers.conv2d_transpose(h, 64*4, 5, stride=2, padding='SAME', activation_fn=None, weights_initializer=initializer)
    h = layers.batch_norm(h, activation_fn=lrelu)
    # h = conv_cond_concat(h, yb)

    h = layers.conv2d_transpose(h, 64*2, 5, stride=2, padding='SAME', activation_fn=None, weights_initializer=initializer)
    h = layers.batch_norm(h, activation_fn=lrelu)
    # h = conv_cond_concat(h, yb)

    h = layers.conv2d_transpose(h, 64*1, 5, stride=2,padding='SAME', activation_fn=None, weights_initializer=initializer)
    h = layers.batch_norm(h, activation_fn=lrelu)

    # h = conv_cond_concat(h, yb)
    h = layers.conv2d_transpose(h, 3, 5, stride=2, padding='SAME', activation_fn=tf.nn.tanh, weights_initializer=initializer)
    return h

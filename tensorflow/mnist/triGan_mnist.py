from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
GPUID = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)
import numpy as np
import tensorflow as tf
import scipy.ndimage.interpolation
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from model_mnist_utils import encoder, discriminator
import scipy.io as sio

""" parameters """
n_epoch = 200
batch_size  = 100
dataset_size = 55000
input_dim = 784
latent_dim = 2
eps_dim = 2

mb_size = 32
X_dim = 784
z_dim = 64
y_dim = 10
h_dim = 128
lr = 1e-4
d_steps = 3

#####################################
def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])
def log(x):
    return tf.log(x + 1e-8)
    # return x
""" Create dataset """

# Create X dataset by importing MNIST data

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=False)

# trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

""" data pre-process """
X_labeled = []
num_per_class = 10 # can be changed to different percentage
count = np.zeros(10)
while True:
    tempx, tempy = mnist.train.next_batch(1)
    if np.sum(count) == 10*num_per_class:
        break
    if count[tempy] < num_per_class:
        count[tempy] += 1
        X_labeled.append(tempx)
    else: continue

X_labeled = np.squeeze(np.float32(X_labeled))
# pdb.set_trace()

y_labeled = np.reshape(X_labeled, [-1, 28, 28])
y_labeled = scipy.ndimage.interpolation.rotate(y_labeled, 90, axes=(1, 2))
y_labeled = np.reshape(y_labeled, [-1, 28*28])

X_unlabeled, y_unlabeled = mnist.train.next_batch(50000)
y_unlabeled = np.reshape(X_unlabeled, [-1,28,28])
y_unlabeled = scipy.ndimage.interpolation.rotate(y_unlabeled, 90, axes=(1, 2))
y_unlabeled = np.reshape(y_unlabeled, [-1, 28*28])

##################
def sample_XY(X,Y, size):
    start_idx = np.random.randint(0, X.shape[0]-size)
    return X[start_idx:start_idx+size], Y[start_idx:start_idx+size]

def sample_X(X, size):
    start_idx = np.random.randint(0, X.shape[0]-size)
    return X[start_idx:start_idx+size]

def sample_Y(Y, size):
    start_idx = np.random.randint(0, Y.shape[0]-size)
    return Y[start_idx:start_idx+size]

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig
#####################

""" Networks """
def generative_Y2X(y):
    with tf.variable_scope("Y2X"):
        h = encoder(y, 28*28)
    return h
def generative_X2Y(x):
    with tf.variable_scope("X2Y"):
        h = encoder(x, 28*28)
    return h

def data_network_1(x, y):
    """Approximate z log data density."""
    with tf.variable_scope('D1'):
        d = discriminator(x, y)
    return tf.squeeze(d, squeeze_dims=[1])

def data_network_2(x, y):
    """Approximate z log data density."""
    with tf.variable_scope('D2'):
        d = discriminator(x, y)
    return tf.squeeze(d, squeeze_dims=[1])


""" Construct model and training ops """
tf.reset_default_graph()

X_p = tf.placeholder(tf.float32, shape=[None, X_dim])
y_p = tf.placeholder(tf.float32, shape=[None, X_dim])
X_u = tf.placeholder(tf.float32, shape=[None, X_dim])
y_u = tf.placeholder(tf.float32, shape=[None, X_dim])
z = tf.placeholder(tf.float32, shape=[None, z_dim])

# Discriminator A
y_gen = generative_X2Y(X_u)
X_gen = generative_Y2X(y_u)
D1_real = data_network_1(X_p, y_p)
D1_fake_y = data_network_1(X_u, y_gen)
D1_fake_X = data_network_1(X_gen, y_u)

# Discriminator B
D2_real = data_network_2(X_u, y_gen)
D2_fake = data_network_2(X_gen, y_u)

# Discriminator loss
L_D1 = -tf.reduce_mean(log(D1_real) + log(1 - D1_fake_y) + log(1 - D1_fake_X))
L_D2 = -tf.reduce_mean(log(D2_real) + log(1 - D2_fake))

D_loss = L_D1 + L_D2

# Generator loss
L_G1 = -tf.reduce_mean(log(D1_fake_y) + log(1-D2_real))
L_G2 = -tf.reduce_mean(log(D1_fake_X) + log(D2_fake))

G_loss = L_G1 + L_G2

# Solvers

gvar1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "Y2X")
gvar2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "X2Y")
dvars1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "D1")
dvars2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "D2")
opt = tf.train.AdamOptimizer(lr, beta1=0.5)

D_solver = opt.minimize(D_loss, var_list=dvars1 + dvars2)
G_solver = opt.minimize(G_loss, var_list=gvar1 + gvar2)

# Call this after declaring all tf.Variables.
saver = tf.train.Saver()

""" training """
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

if not os.path.exists('out_semi_gan_1000/'):
    os.makedirs('out_semi_gan_1000/')


i = 0
init = tf.global_variables_initializer()

for it in range(10000000):
    # Sample data from both domains

    # for _ in range(3):
    # X_p_mb, y_p_mb = sample_XY(X_labeled,y_labeled, mb_size)
    # X_u_mb = sample_X(X_unlabeled, mb_size)
    # y_u_mb = sample_Y(y_unlabeled, mb_size)
    # z_sample = sample_Z(mb_size, z_dim)
    X_p_mb, y_p_mb = sample_XY(X_labeled, y_labeled, mb_size)
    X_u_mb = sample_X(X_unlabeled, mb_size)
    y_u_mb = sample_Y(y_unlabeled, mb_size)
    z_sample = sample_Z(mb_size, z_dim)
    for k in range(5):
        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X_p: X_p_mb, y_p: y_p_mb, X_u: X_u_mb, y_u: y_u_mb})
    for j in range(1):
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={X_p: X_p_mb, y_p: y_p_mb, X_u: X_u_mb, y_u: y_u_mb})

    if it % 100 == 0:
        print('Iter: {}; D_loss: {:.4}; G_loss: {:.4}'.format(it, D_loss_curr, G_loss_curr))

        # input_A = sample_X(X_unlabeled, size=4)
        # input_B = sample_X(y_unlabeled, size=4)
        #
        # samples_A = sess.run(X_gen, feed_dict={y_u: input_B})
        # samples_B = sess.run(y_gen, feed_dict={X_u: input_A})
        #
        # # The resulting image sample would be in 4 rows:
        # # row 1: real data from domain A, row 2 is its domain B translation
        # # row 3: real data from domain B, row 4 is its domain A translation
        # samples = np.vstack([input_A, samples_B, input_B, samples_A])
        #
        # fig = plot(samples)
        # plt.savefig('out_semi_gan_1000/{}.png'
        #             .format(str(i).zfill(3)), bbox_inches='tight')
        # i += 1
        # plt.close(fig)
    if it % 200 == 0:
        # pdb.set_trace()
        input_A, label = mnist.test.next_batch(10000)
        input_B = np.reshape(input_A, [-1, 28, 28])
        input_B = scipy.ndimage.interpolation.rotate(input_B, 90, axes=(1, 2))
        input_B = np.reshape(input_B, [-1, 28*28])
        samples_A = sess.run(X_gen, feed_dict={y_u: input_B})
        samples_A = np.reshape(samples_A, [-1, 28, 28])
        samples_B = sess.run(y_gen, feed_dict={X_u: input_A})
        samples_B = np.reshape(samples_B, [-1, 28, 28])
        samples_B = scipy.ndimage.interpolation.rotate(samples_B, 270, axes=(1, 2))
        tmp = np.max(label) + 1
        label = np.uint8(np.eye(tmp)[label])
        del tmp
        sio.savemat('./valid/100/label_%d.mat' % it, {'label': label})
        sio.savemat('./valid/100/sampleB_%d.mat' % it, {'dataB': samples_B})
        sio.savemat('./valid/100/sampleA_%d.mat' % it, {'dataA': samples_A})
        print("finish saving!")

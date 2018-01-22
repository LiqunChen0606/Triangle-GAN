from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
GPUID = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)
import numpy as np
import tensorflow as tf
import scipy.ndimage.interpolation
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from model_celebA_utils import encoder1, encoder2, discriminator

import scipy.io as sio
import pdb
import h5py
import json
import time
import cPickle

""" parameters """
n_epochs = 10
dataset_size = 50000
mb_size = 64
# X_dim = ()
lr = 1e-4
Z_dim = 100
Y_dim = 40
#####################################
def log(x):
    return tf.log(x + 1e-8)

""" data pre-process """
hdf5_root = '/home/lqchen/work/pixel-cnn-3/data/CelebA/'
f = h5py.File('%sceleba_64.hdf5' % hdf5_root)
Images = np.float32(f['features']) / 127.5 - 1.
feature_data = scipy.io.loadmat('/home/lqchen/work/triGan/celebA/celebA_tag_feats.mat')

tag_feats_all = np.float32(feature_data['feats'])
tag_feats = tag_feats_all[:162770]
tag_feats_val = tag_feats_all[162770: 182637]
tag_feats_test = tag_feats_all[182637:]
sio.savemat('./evaluation/10/tag_feats_val.mat', {'feats_val': tag_feats_val})
sio.savemat('./evaluation/10/tag_feats_test.mat', {'feats_test': tag_feats_test})
del feature_data, tag_feats_all

Images_all = np.transpose(Images, [0,2,3,1])
Images = Images_all[:162770]
Images_val = Images_all[162770: 182637]
Images_test = Images_all[182637:]
del Images_all

num_train = Images.shape[0]
num_val = Images_val.shape[0]
num_test = Images_test.shape[0]

""" tag name"""
tag_names = []
with open('./celebA_tag_names.txt', 'rb') as f:
    for line in f:
        tag_names = line.strip().split(",")

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
def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])
def plot(samples):
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample)
    return fig
#####################

""" Networks """
def generative_Y2X(y, z, reuse = None):
    with tf.variable_scope("Y2X", reuse=reuse):
        h = encoder2(y, z)
    return h


def generative_X2Y(x, reuse=None):
    with tf.variable_scope("X2Y", reuse=reuse):
        h = encoder1(x)
    return h


def data_network_1(x, y, reuse=None):
    """Approximate z log data density."""
    with tf.variable_scope('D1', reuse=reuse):
        d = discriminator(x, y)
    return tf.squeeze(d, squeeze_dims=[1])


def data_network_2(x, y, reuse=None):
    """Approximate z log data density."""
    with tf.variable_scope('D2', reuse=reuse):
        d = discriminator(x, y)
    return tf.squeeze(d, squeeze_dims=[1])


""" Construct model and training ops """
# tf.reset_default_graph()

X_p = tf.placeholder(tf.float32, shape=[mb_size, 64, 64, 3])
y_p = tf.placeholder(tf.float32, shape=[mb_size, Y_dim])
X_u = tf.placeholder(tf.float32, shape=[mb_size, 64, 64, 3])
y_u = tf.placeholder(tf.float32, shape=[mb_size, Y_dim])
z = tf.placeholder(tf.float32, shape=[mb_size, Z_dim])

# Discriminator A
y_gen = generative_X2Y(X_u)
X_gen = generative_Y2X(y_u, z)

D1_real = data_network_1(X_p, y_p)
D1_fake_y = data_network_1(X_u, y_gen, reuse=True)
D1_fake_X = data_network_1(X_gen, y_u, reuse=True)

# Discriminator B
D2_real = data_network_2(X_u, y_gen)
D2_fake = data_network_2(X_gen, y_u, reuse=True)

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
gvar2= tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "X2Y")
dvars1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "D1")
dvars2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "D2")
opt = tf.train.AdamOptimizer(lr, beta1=0.5)

D_solver = opt.minimize(D_loss, var_list = dvars1 + dvars2)
G_solver = opt.minimize(G_loss, var_list = gvar1 + gvar2)

# Call this after declaring all tf.Variables.
saver = tf.train.Saver()

""" training """
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess = tf.Session(config=config)

ii = 0
init = tf.global_variables_initializer()
sess.run(init)
# Load pretrained Model
try:
    saver.restore(sess=sess, save_path="./model/model_trigan_CelebA.ckpt")
    print("\n--------model restored--------\n")
except:
    print("\n--------model Not restored--------\n")
    pass

zz = sample_Z(mb_size, Z_dim)
disc_steps = 1
gen_steps = 2
paired_data_num = np.int32(0.1 * num_train) # can change to 20%, 0.1%...
paired_data, paired_tag = sample_XY(Images, tag_feats, paired_data_num)
for it in range(n_epochs):

    for idx in range(0, num_train // mb_size):
        X_p_mb, y_p_mb = sample_XY(paired_data, paired_tag, mb_size)

        X_u_mb = Images[idx*mb_size : (idx + 1) * mb_size]
        y_u_mb = tag_feats[idx*mb_size: (idx + 1) * mb_size]

        z_sample = sample_Z(mb_size, Z_dim)
        for k in range(disc_steps):
            _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X_p: X_p_mb, y_p: y_p_mb, X_u: X_u_mb, y_u: y_u_mb, z: z_sample})
        for j in range(gen_steps):
            _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={X_p: X_p_mb, y_p: y_p_mb, X_u: X_u_mb, y_u: y_u_mb, z: z_sample})

        if idx % 200 == 0:
            print('epoch: {}; iter: {}; D_loss: {:.4}; G_loss: {:.4}'.format(it, idx, D_loss_curr, G_loss_curr))

            input_B = sample_X(tag_feats, size=8)
            input_B = np.repeat(input_B, 8, axis=0)
            zz = sample_Z(8, Z_dim)
            zz = np.tile(zz, (8,1))

            samples_A = sess.run(X_gen, feed_dict={y_u: input_B, z: zz})

            # The resulting image sample would be in 4 rows:
            # row 1: real data from domain A, row 2 is its domain B translation
            # row 3: real data from domain B, row 4 is its domain A translation

            tag_feats_s = input_B
            f = open('./semi_10/tags_{}.txt'.format(idx), 'w')
            for i in range(tag_feats_s.shape[0]):
                for j in range(tag_feats_s.shape[1]):
                    if tag_feats_s[i][j] == 1:
                        f.write(tag_names[j]+", ")
                f.write("\n\n")
            f.close()
            samples_A = (samples_A + 1.) / 2.
            fig = plot(samples_A)
            plt.savefig('./semi_10/{}.png'
                        .format(str(idx).zfill(3)), bbox_inches='tight')
            ii += 1
            plt.close(fig)
            saver.save(sess, './model/model_trigan_celeb_eval_10.ckpt')

    # if it % 10 == 0:
    #     print('Iter: {}; D_loss: {:.4}; G_loss: {:.4}'.format(it, D_loss_curr, G_loss_curr))
    
    #     input_B = sample_X(tag_feats, size=8)
    #     input_B = np.repeat(input_B, 8, axis=0)
    #     zz = sample_Z(8, Z_dim)
    #     zz = np.repeat(zz, 8, axis=0)
    #     pdb.set_trace()
    #     samples_A = sess.run(X_gen, feed_dict={y_u: input_B, z: zz})
    
    #     # The resulting image sample would be in 4 rows:
    #     # row 1: real data from domain A, row 2 is its domain B translation
    #     # row 3: real data from domain B, row 4 is its domain A translation
    
    #     tag_feats_s = input_B
    #     f = open('./out_semi_gan_few/tags_{}.txt'.format(ii), 'w')
    #     for i in range(tag_feats_s.shape[0]):
    #         for j in range(tag_feats_s.shape[1]):
    #             if tag_feats_s[i][j] == 1:
    #                 f.write(tag_names[j]+", ")
    #         f.write("\n\n")
    #     f.close()
    #     samples_A = (samples_A + 1.) / 2.
    #     fig = plot(samples_A)
    #     plt.savefig('out_semi_gan_few/{}.png'
    #                 .format(str(ii).zfill(3)), bbox_inches='tight')
    #     ii += 1
    #     plt.close(fig)

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
# from model_py35 import encoder1, encoder2, discriminator, SN_discriminator
from model_fancyCelebA_utils import encoder1, encoder2, discriminator, SN_discriminator
import scipy.io as sio
import pdb
import h5py
import json
import time
# import cPickle
from eval import diff
from sklearn.metrics import roc_auc_score
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
    return tf.log(x + 1e-10)

""" data pre-process """
hdf5_root = '/home/lqchen/work/pixel-cnn-3/data/CelebA/'
f = h5py.File('%sceleba_64.hdf5' % hdf5_root)
Images = np.float32(f['features']) / 127.5 - 1.
feature_data = scipy.io.loadmat('/home/lqchen/work/triGan/celebA/celebA_tag_feats.mat')

tag_feats_all = np.float32(feature_data['feats'])
tag_feats = tag_feats_all[:162770]
tag_feats_val = tag_feats_all[162770: 182637]
tag_feats_test = tag_feats_all[182637:]
# sio.savemat('./evaluation/tag_feats_val.mat', {'feats_val': tag_feats_val})
# sio.savemat('./evaluation/tag_feats_test.mat', {'feats_test': tag_feats_test})
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
        # tag_names = line.strip().split(",")
        tag_names = line.decode('utf8').strip().split(",")
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
def generative_Y2X(y, z, reuse=None):
    with tf.variable_scope("Y2X", reuse=reuse):
        h = encoder2(y, z)
    return h
def generative_X2Y(x, reuse=None):
    with tf.variable_scope("X2Y", reuse=reuse):
        h = encoder1(x)
    return h

def data_network_1(x, y, reuse=None):
    """Approximate z log data density."""
    with tf.variable_scope('D1', reuse=reuse) as scope:
        # d = discriminator(x, y)
        d = SN_discriminator(x, y)
    return d #tf.squeeze(d, squeeze_dims=[1])

def data_network_2(x, y, reuse=None):
    """Approximate z log data density."""
    with tf.variable_scope('D2', reuse=reuse) as scope:
        # d = discriminator(x, y)
        d = SN_discriminator(x, y)
    return d


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
D1_fake_y = data_network_1(X_u, y_gen, True)
D1_fake_X = data_network_1(X_gen, y_u, True)

# Discriminator B
D2_real = data_network_2(X_u, y_gen)
D2_fake = data_network_2(X_gen, y_u, True)

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
    saver.restore(sess=sess, save_path="./model/model_trigan_celeb_64.ckpt")
    print("\n--------model restored--------\n")
except:
    print("\n--------model Not restored--------\n")
    pass

# zz = sample_Z(mb_size, Z_dim)
disc_steps = 1
gen_steps = 1
for it in range(n_epochs):
    # pdb.set_trace()
    arr = np.random.permutation(num_train)
    Images = Images[arr]
    tag_feats = tag_feats[arr]

    for idx in range(0, num_train // mb_size):
        # X_p_mb, y_p_mb = sample_XY(Images, tag_feats, mb_size)

        X_p_mb = Images[idx*mb_size : (idx + 1) * mb_size]
        y_p_mb = tag_feats[idx*mb_size: (idx + 1) * mb_size]
        X_u_mb = X_p_mb
        y_u_mb = y_p_mb
        z_sample = sample_Z(mb_size, Z_dim)
        for k in range(disc_steps):
            _, D_loss_curr = sess.run([D_solver, D_loss],
                feed_dict={X_p: X_p_mb, y_p: y_p_mb, X_u: X_u_mb, y_u: y_u_mb, z: z_sample})
        for j in range(gen_steps):
            _, G_loss_curr = sess.run([G_solver, G_loss],
                feed_dict={X_p: X_p_mb, y_p: y_p_mb, X_u: X_u_mb, y_u: y_u_mb, z: z_sample})

        if idx % 100 == 0:
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
            f = open('./64/tags_{}.txt'.format(idx), 'w')
            for i in range(tag_feats_s.shape[0]):
                for j in range(tag_feats_s.shape[1]):
                    if tag_feats_s[i][j] == 1:
                        f.write(tag_names[j]+", ")
                f.write("\n\n")
            f.close()
            samples_A = (samples_A + 1.) / 2.
            fig = plot(samples_A)
            plt.savefig('./64/{}.png'
                        .format(str(idx).zfill(3)), bbox_inches='tight')
            ii += 1
            plt.close(fig)
            saver.save(sess, './model/model_trigan_celeb_64.ckpt')

    # if it % 1 == 0:
    #     # Test on the validation dataset
    #
    #     count = 0
    #     val_auc = 0
    #     val_score = np.zeros((1,40))
    #     for j in range(0, num_val // mb_size):
    #         val_x = Images_val[j*mb_size : (j + 1) * mb_size]
    #         y_true = tag_feats_val[j*mb_size : (j + 1) * mb_size]
    #         y_scores = sess.run(y_gen, feed_dict={X_u: val_x})
    #         y_scores = np.squeeze(y_scores)
    #         val_score = np.concatenate([val_score, y_scores], 0)
    #         # val_auc += roc_auc_score(y_true, y_scores)
    #         val_auc += diff(y_scores, y_true)
    #         count += 1
    #     val_auc /= count
    #     del val_x
    #     val_score = val_score[1:]
    #     # Test on the test dataset
    #     count = 0
    #     test_auc = 0
    #     test_score = np.zeros((1,40))
    #     for j in range(0, num_test // mb_size):
    #         test_x = Images_test[j*mb_size : (j + 1) * mb_size]
    #         y_true = tag_feats_test[j*mb_size : (j + 1) * mb_size]
    #         y_scores = sess.run(y_gen, feed_dict={X_u: test_x})
    #         y_scores = np.squeeze(y_scores)
    #         test_score = np.concatenate([test_score, y_scores], 0)
    #         test_auc += diff(y_scores, y_true)
    #         count += 1
    #     test_auc /= count
    #     del test_x
    #     test_score = test_score[1:]
    #     sio.savemat('./evaluation/full/val_score_%d.mat' % it, {'val_score': val_score})
    #     sio.savemat('./evaluation/full/test_score_%d.mat' % it, {'test_score': test_score})
    #     print('epoch: {};  validation auc: {:.4}; test auc: {:.4}'.format(it, val_auc, test_auc))
    # if it % 1 == 0:
    #     print('Iter: {}; D_loss: {:.4}; G_loss: {:.4}'.format(it, D_loss_curr, G_loss_curr))
    #
    #     input_B = sample_X(tag_feats, size=8)
    #     input_B = np.repeat(input_B, 8, axis=0)
    #     zz = sample_Z(8, Z_dim)
    #     zz = np.repeat(zz, 8, axis=0)
    #     pdb.set_trace()
    #     samples_A = sess.run(X_gen, feed_dict={y_u: input_B, z: zz})
    #
    #     # The resulting image sample would be in 4 rows:
    #     # row 1: real data from domain A, row 2 is its domain B translation
    #     # row 3: real data from domain B, row 4 is its domain A translation
    #
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


    # if it % 200 == 0:
    #     # pdb.set_trace()
    #     input_A, label = mnist.test.next_batch(10000)
    #     input_B = np.reshape(input_A, [-1, 28, 28])
    #     input_B = scipy.ndimage.interpolation.rotate(input_B, 90, axes=(1, 2))
    #     input_B = np.reshape(input_B, [-1, 28*28])
    #     samples_A = sess.run(X_gen, feed_dict={y_u: input_B})
    #     samples_A = np.reshape(samples_A, [-1, 28, 28])
    #     samples_B = sess.run(y_gen, feed_dict={X_u: input_A})
    #     samples_B = np.reshape(samples_B, [-1, 28, 28])
    #     samples_B = scipy.ndimage.interpolation.rotate(samples_B, 270, axes=(1, 2))
    #     tmp = np.max(label) + 1
    #     label = np.uint8(np.eye(tmp)[label])
    #     del tmp
    #     sio.savemat('./valid_10000/label_%d.mat' % it, {'label': label})
    #     sio.savemat('./valid_10000/sampleB_%d.mat' % it, {'dataB': samples_B})
    #     sio.savemat('./valid_10000/sampleA_%d.mat' % it, {'dataA': samples_A})
    #     print("finish saving!")

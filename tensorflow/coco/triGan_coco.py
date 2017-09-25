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
# from model_coco_utils import encoder1, encoder2, discriminator
from model_fancyCOCO_utils import encoder1, encoder2, discriminator
import scipy.io as sio
import pdb
import h5py
import json
import time
import cPickle
from eval import diff
""" parameters """
dataset_size = 50000
mb_size = 64
X_dim = 784
lr = 1e-4
Z_dim = 128
Y_dim = 999
n_epochs = 250
#####################################
def log(x):
    return tf.log(x + 1e-8)
    # return x
""" Create dataset """
img_size = 64
hdf5_root = '/media/lqchen/MyFiles/Data/coco/'
f = h5py.File('%scoco_img_%d.h5'%(hdf5_root,img_size))
Images = np.float32(f['images']) / 127.5 - 1.
# feature_data = scipy.io.loadmat('/media/lqchen/MyFiles/Data/coco/coco_tag_feats.mat')
feature_data = h5py.File('/media/lqchen/MyFiles/Data/coco/coco_tag_feat_binary.hdf5')
tag_feats = np.float32(feature_data['feats'])
Images = np.transpose(Images, [0,2,3,1])
del feature_data

## index for train validation and test
with open ('train_list', 'rb') as fp: train_id = cPickle.load(fp)
with open ('val_list', 'rb') as fp: val_id = cPickle.load(fp)
with open ('test_list', 'rb') as fp: test_id = cPickle.load(fp)
fp.close()

val_Images = Images[val_id]
val_tags = tag_feats[val_id]
num_val = len(val_id)
test_Images = Images[test_id]
test_tags = tag_feats[test_id]
num_test = len(test_id)
Images = Images[train_id]
tag_feats = tag_feats[train_id]
num_train = len(train_id)
del train_id, val_id, test_id
sio.savemat('./evaluation/50/tag_feats_val.mat', {'feats_val': val_tags})
sio.savemat('./evaluation/50/tag_feats_test.mat', {'feats_test': test_tags})
# Create X dataset by importing MNIST data
""" data pre-process """

""" tag name"""
x = cPickle.load(open("/media/lqchen/MyFiles/Data/coco/coco_tag_vocab.p","rb"))
wordtoix, ixtoword = x[0],x[1]
del x

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
def generative_Y2X(y, z):
    with tf.variable_scope("Y2X"):
        h = encoder2(y, z)
    return h
def generative_X2Y(x):
    with tf.variable_scope("X2Y"):
        h = encoder1(x)
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

X_p = tf.placeholder(tf.float32, shape=[mb_size, 64, 64, 3])
y_p = tf.placeholder(tf.float32, shape=[mb_size, Y_dim])
X_u = tf.placeholder(tf.float32, shape=[mb_size, 64, 64, 3])
y_u = tf.placeholder(tf.float32, shape=[mb_size, Y_dim])
z = tf.placeholder(tf.float32, shape=[mb_size, Z_dim])

# Discriminator A
y_gen = generative_X2Y(X_u)
X_gen = generative_Y2X(y_u, z)
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
# try:
#     saver.restore(sess=sess, save_path="./model/50/model_trigan_coco_test_50.ckpt")
#     print("\n--------model restored--------\n")
# except:
#     print("\n--------model Not restored--------\n")
#     pass
disc_steps = 1
gen_steps = 1
paired_data_num = np.int32(0.5 * num_train) # change this to different percentage
paired_data, paired_tag = sample_XY(Images, tag_feats, paired_data_num)
for it in range(n_epochs):
    arr = np.random.permutation(num_train)
    Images = Images[arr]
    arr = np.random.permutation(num_train)
    tag_feats = tag_feats[arr]
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
            print('Epoch: {}; Iter: {}; D_loss: {:.4}; G_loss: {:.4}'.format(it, idx, D_loss_curr, G_loss_curr))
            print("start plotting")
            # input_A = sample_X(X_unlabeled, size=16)
            input_B = sample_Y(tag_feats, size=mb_size)
            z_sample = sample_Z(mb_size, Z_dim)
            samples_A = sess.run(X_gen, feed_dict={y_u: input_B, z: z_sample})
            # samples_B = sess.run(y_gen, feed_dict={X_u: input_A})

            # The resulting image sample would be in 4 rows:
            # row 1: real data from domain A, row 2 is its domain B translation
            # row 3: real data from domain B, row 4 is its domain A translation
            # samples = np.vstack([samples_A])

            samples_A = (samples_A + 1) / 2.
            fig = plot(samples_A)
            plt.savefig('./result/50/{}.png'
                        .format(str(ii).zfill(3)), bbox_inches='tight')
            plt.close(fig)

            tag_feats_s = input_B
            f = open('./result/50/tags_{}.txt'.format(ii), 'w')
            for i in range(tag_feats_s.shape[0]):
                for j in range(tag_feats_s.shape[1]):
                    if tag_feats_s[i][j] == 1.:
                        f.write(ixtoword[j]+", ")
                f.write("\n\n")
            f.close()

            ii += 1

            saver.save(sess, './model/50/model_trigan_coco_test_50.ckpt')
            print("Finish saving")
    if it % 1 == 0:
        # Test on the validation dataset

        count = 0
        val_auc = 0
        val_score = np.zeros((1,999))
        for j in range(0, num_val // mb_size):
            val_x = val_Images[j*mb_size : (j + 1) * mb_size]
            y_true = val_tags[j*mb_size : (j + 1) * mb_size]
            y_scores = sess.run(y_gen, feed_dict={X_u: val_x})
            y_scores = np.squeeze(y_scores)
            val_score = np.concatenate([val_score, y_scores], 0)
            # val_auc += roc_auc_score(y_true, y_scores)
            val_auc += diff(y_scores, y_true)
            count += 1
        val_auc /= count
        del val_x
        val_score = val_score[1:]
        # Test on the test dataset
        count = 0
        test_auc = 0
        test_score = np.zeros((1,999))
        for j in range(0, num_test // mb_size):
            test_x = test_Images[j*mb_size : (j + 1) * mb_size]
            y_true = test_tags[j*mb_size : (j + 1) * mb_size]
            y_scores = sess.run(y_gen, feed_dict={X_u: test_x})
            y_scores = np.squeeze(y_scores)
            test_score = np.concatenate([test_score, y_scores], 0)
            test_auc += diff(y_scores, y_true)
            count += 1
        test_auc /= count
        del test_x
        test_score = test_score[1:]
        sio.savemat('./evaluation/50/val_score_%d.mat' % it, {'val_score': val_score})
        sio.savemat('./evaluation/50/test_score_%d.mat' % it, {'test_score': test_score})
        print('epoch: {};  validation auc: {:.4}; test auc: {:.4}'.format(it, val_auc, test_auc))

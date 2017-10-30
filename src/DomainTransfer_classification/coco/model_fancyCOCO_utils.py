import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import os

batch_size = 64

# z_dim = 512         # Noise dimension
image_size = 64     # 64 x 64
c_dim = 3           # for rgb
t_dim = 999         # text feature dimension

def encoder1(input_images, t_txt=None, is_train=True):
    """ 64x64 + (txt) --> real/fake """
    # https://github.com/hanzhanggit/StackGAN/blob/master/stageI/model.py
    # Discriminator with ResNet : line 197 https://github.com/reedscot/icml2016/blob/master/main_cls.lua
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init=tf.random_normal_initializer(1., 0.02)
    df_dim = 64  # 64 for flower, 196 for MSCOCO
    s = 64 # output image size [64]
    s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)


    net_in = InputLayer(input_images, name='e_input/images')
    net_h0 = Conv2d(net_in, df_dim, (4, 4), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2),
            padding='SAME', W_init=w_init, name='e_h0/conv2d')

    net_h1 = Conv2d(net_h0, df_dim*2, (4, 4), (2, 2), act=None,
            padding='SAME', W_init=w_init, b_init=None, name='e_h1/conv2d')
    net_h1 = BatchNormLayer(net_h1, act=lambda x: tl.act.lrelu(x, 0.2),
            is_train=is_train, gamma_init=gamma_init, name='e_h1/batchnorm')
    net_h2 = Conv2d(net_h1, df_dim*4, (4, 4), (2, 2), act=None,
            padding='SAME', W_init=w_init, b_init=None, name='e_h2/conv2d')
    net_h2 = BatchNormLayer(net_h2, act=lambda x: tl.act.lrelu(x, 0.2),
            is_train=is_train, gamma_init=gamma_init, name='e_h2/batchnorm')
    net_h3 = Conv2d(net_h2, df_dim*8, (4, 4), (2, 2), act=None,
            padding='SAME', W_init=w_init, b_init=None, name='e_h3/conv2d')
    net_h3 = BatchNormLayer(net_h3, #act=lambda x: tl.act.lrelu(x, 0.2),
            is_train=is_train, gamma_init=gamma_init, name='e_h3/batchnorm')

    net = Conv2d(net_h3, df_dim*2, (1, 1), (1, 1), act=None,
            padding='VALID', W_init=w_init, b_init=None, name='e_h4_res/conv2d')
    net = BatchNormLayer(net, act=lambda x: tl.act.lrelu(x, 0.2),
            is_train=is_train, gamma_init=gamma_init, name='e_h4_res/batchnorm')
    net = Conv2d(net, df_dim*2, (3, 3), (1, 1), act=None,
            padding='SAME', W_init=w_init, b_init=None, name='e_h4_res/conv2d2')
    net = BatchNormLayer(net, act=lambda x: tl.act.lrelu(x, 0.2),
            is_train=is_train, gamma_init=gamma_init, name='e_h4_res/batchnorm2')
    net = Conv2d(net, df_dim*8, (3, 3), (1, 1), act=None,
            padding='SAME', W_init=w_init, b_init=None, name='e_h4_res/conv2d3')
    net = BatchNormLayer(net, #act=lambda x: tl.act.lrelu(x, 0.2),
            is_train=is_train, gamma_init=gamma_init, name='e_h4_res/batchnorm3')
    net_h4 = ElementwiseLayer(layer=[net_h3, net], combine_fn=tf.add, name='e_h4/add')
    net_h4.outputs = tl.act.lrelu(net_h4.outputs, 0.2)

    if t_txt is not None:
        net_txt = InputLayer(t_txt, name='e_input_txt')
        net_txt = DenseLayer(net_txt, n_units=t_dim,
               act=lambda x: tl.act.lrelu(x, 0.2),
               W_init=w_init, name='e_reduce_txt/dense')
        net_txt = ExpandDimsLayer(net_txt, 1, name='e_txt/expanddim1')
        net_txt = ExpandDimsLayer(net_txt, 1, name='e_txt/expanddim2')
        net_txt = TileLayer(net_txt, [1, 4, 4, 1], name='e_txt/tile')
        net_h4_concat = ConcatLayer([net_h4, net_txt], concat_dim=3, name='e_h3_concat')
        # 243 (ndf*8 + 128 or 256) x 4 x 4
        net_h4 = Conv2d(net_h4_concat, df_dim*8, (1, 1), (1, 1),
                padding='VALID', W_init=w_init, b_init=None, name='e_h3/conv2d_2')
        net_h4 = BatchNormLayer(net_h4, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='e_h3/batch_norm_2')

    net_ho = Conv2d(net_h4, t_dim, (s16, s16), (s16, s16), padding='VALID', W_init=w_init, name='e_ho/conv2d')
    # 1 x 1 x 1
    # net_ho = FlattenLayer(net_h4, name='d_ho/flatten')
    logits = net_ho.outputs
    net_ho.outputs = tf.nn.sigmoid(net_ho.outputs)
    return net_ho.outputs

## default g1, d1 ==============================================================
def encoder2(t_txt, input_z, is_train=True, batch_size=batch_size):
    """ z + (txt) --> 64x64 """
    # https://github.com/hanzhanggit/StackGAN/blob/master/stageI/model.py
    s = image_size # output image size [64]
    s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
    gf_dim = 128

    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    net_in = InputLayer(input_z, name='g_inputz')

    if t_txt is not None:
        net_txt = InputLayer(t_txt, name='g_input_txt')
        net_txt = DenseLayer(net_txt, n_units=t_dim,
            act=lambda x: tl.act.lrelu(x, 0.2), W_init=w_init, name='g_reduce_text/dense')
        net_in = ConcatLayer([net_in, net_txt], concat_dim=1, name='g_concat_z_txt')

    net_h0 = DenseLayer(net_in, gf_dim*8*s16*s16, act=tf.identity,
            W_init=w_init, b_init=None, name='g_h0/dense')
    net_h0 = BatchNormLayer(net_h0,  #act=tf.nn.relu,
            is_train=is_train, gamma_init=gamma_init, name='g_h0/batch_norm')
    net_h0 = ReshapeLayer(net_h0, [-1, s16, s16, gf_dim*8], name='g_h0/reshape')

    net = Conv2d(net_h0, gf_dim*2, (1, 1), (1, 1),
            padding='VALID', act=None, W_init=w_init, b_init=None, name='g_h1_res/conv2d')
    net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train,
            gamma_init=gamma_init, name='g_h1_res/batch_norm')
    net = Conv2d(net, gf_dim*2, (3, 3), (1, 1),
            padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h1_res/conv2d2')
    net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train,
            gamma_init=gamma_init, name='g_h1_res/batch_norm2')
    net = Conv2d(net, gf_dim*8, (3, 3), (1, 1),
            padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h1_res/conv2d3')
    net = BatchNormLayer(net, # act=tf.nn.relu,
            is_train=is_train, gamma_init=gamma_init, name='g_h1_res/batch_norm3')
    net_h1 = ElementwiseLayer(layer=[net_h0, net], combine_fn=tf.add, name='g_h1_res/add')
    net_h1.outputs = tf.nn.relu(net_h1.outputs)

    # Note: you can also use DeConv2d to replace UpSampling2dLayer and Conv2d
    # net_h2 = DeConv2d(net_h1, gf_dim*4, (4, 4), out_size=(s8, s8), strides=(2, 2),
    #         padding='SAME', batch_size=batch_size, act=None, W_init=w_init, b_init=b_init, name='g_h2/decon2d')
    net_h2 = UpSampling2dLayer(net_h1, size=[s8, s8], is_scale=False, method=1,
            align_corners=False, name='g_h2/upsample2d')
    net_h2 = Conv2d(net_h2, gf_dim*4, (3, 3), (1, 1),
            padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h2/conv2d')
    net_h2 = BatchNormLayer(net_h2,# act=tf.nn.relu,
            is_train=is_train, gamma_init=gamma_init, name='g_h2/batch_norm')

    net = Conv2d(net_h2, gf_dim, (1, 1), (1, 1),
            padding='VALID', act=None, W_init=w_init, b_init=None, name='g_h3_res/conv2d')
    net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train,
            gamma_init=gamma_init, name='g_h3_res/batch_norm')
    net = Conv2d(net, gf_dim, (3, 3), (1, 1),
            padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h3_res/conv2d2')
    net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train,
            gamma_init=gamma_init, name='g_h3_res/batch_norm2')
    net = Conv2d(net, gf_dim*4, (3, 3), (1, 1),
            padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h3_res/conv2d3')
    net = BatchNormLayer(net, #act=tf.nn.relu,
            is_train=is_train, gamma_init=gamma_init, name='g_h3_res/batch_norm3')
    net_h3 = ElementwiseLayer(layer=[net_h2, net], combine_fn=tf.add, name='g_h3/add')
    net_h3.outputs = tf.nn.relu(net_h3.outputs)

    # net_h4 = DeConv2d(net_h3, gf_dim*2, (4, 4), out_size=(s4, s4), strides=(2, 2),
    #         padding='SAME', batch_size=batch_size, act=None, W_init=w_init, b_init=b_init, name='g_h4/decon2d'),
    net_h4 = UpSampling2dLayer(net_h3, size=[s4, s4], is_scale=False, method=1,
            align_corners=False, name='g_h4/upsample2d')
    net_h4 = Conv2d(net_h4, gf_dim*2, (3, 3), (1, 1),
            padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h4/conv2d')
    net_h4 = BatchNormLayer(net_h4, act=tf.nn.relu,
            is_train=is_train, gamma_init=gamma_init, name='g_h4/batch_norm')

    # net_h5 = DeConv2d(net_h4, gf_dim, (4, 4), out_size=(s2, s2), strides=(2, 2),
    #         padding='SAME', batch_size=batch_size, act=None, W_init=w_init, b_init=b_init, name='g_h5/decon2d')
    net_h5 = UpSampling2dLayer(net_h4, size=[s2, s2], is_scale=False, method=1,
            align_corners=False, name='g_h5/upsample2d')
    net_h5 = Conv2d(net_h5, gf_dim, (3, 3), (1, 1),
            padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h5/conv2d')
    net_h5 = BatchNormLayer(net_h5, act=tf.nn.relu,
            is_train=is_train, gamma_init=gamma_init, name='g_h5/batch_norm')

    # net_ho = DeConv2d(net_h5, c_dim, (4, 4), out_size=(s, s), strides=(2, 2),
    #         padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g_ho/decon2d')
    net_ho = UpSampling2dLayer(net_h5, size=[s, s], is_scale=False, method=1,
            align_corners=False, name='g_ho/upsample2d')
    net_ho = Conv2d(net_ho, c_dim, (3, 3), (1, 1),
            padding='SAME', act=None, W_init=w_init, name='g_ho/conv2d')
    logits = net_ho.outputs
    net_ho.outputs = tf.nn.tanh(net_ho.outputs)
    return net_ho.outputs

# def discriminator(input_images, t_txt=None, is_train=True):
#     """ 64x64 + (txt) --> real/fake """
#     # https://github.com/hanzhanggit/StackGAN/blob/master/stageI/model.py
#     # Discriminator with ResNet : line 197 https://github.com/reedscot/icml2016/blob/master/main_cls.lua
#     w_init = tf.random_normal_initializer(stddev=0.02)
#     gamma_init=tf.random_normal_initializer(1., 0.02)
#     df_dim = 64  # 64 for flower, 196 for MSCOCO
#     s = 64 # output image size [64]
#     s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
#
#     net_in = InputLayer(input_images)
#     net_h0 = Conv2d(net_in, df_dim, (4, 4), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2),
#             padding='SAME', W_init=w_init)
#
#     net_h1 = Conv2d(net_h0, df_dim*2, (4, 4), (2, 2), act=None,
#             padding='SAME', W_init=w_init, b_init=None)
#     net_h1 = BatchNormLayer(net_h1, act=lambda x: tl.act.lrelu(x, 0.2),
#             is_train=is_train, gamma_init=gamma_init)
#     net_h2 = Conv2d(net_h1, df_dim*4, (4, 4), (2, 2), act=None,
#             padding='SAME', W_init=w_init, b_init=None)
#     net_h2 = BatchNormLayer(net_h2, act=lambda x: tl.act.lrelu(x, 0.2),
#             is_train=is_train, gamma_init=gamma_init)
#     net_h3 = Conv2d(net_h2, df_dim*8, (4, 4), (2, 2), act=None,
#             padding='SAME', W_init=w_init, b_init=None)
#     net_h3 = BatchNormLayer(net_h3, #act=lambda x: tl.act.lrelu(x, 0.2),
#             is_train=is_train, gamma_init=gamma_init)
#
#     net = Conv2d(net_h3, df_dim*2, (1, 1), (1, 1), act=None,
#             padding='VALID', W_init=w_init, b_init=None)
#     net = BatchNormLayer(net, act=lambda x: tl.act.lrelu(x, 0.2),
#             is_train=is_train, gamma_init=gamma_init)
#     net = Conv2d(net, df_dim*2, (3, 3), (1, 1), act=None,
#             padding='SAME', W_init=w_init, b_init=None)
#     net = BatchNormLayer(net, act=lambda x: tl.act.lrelu(x, 0.2),
#             is_train=is_train, gamma_init=gamma_init)
#     net = Conv2d(net, df_dim*8, (3, 3), (1, 1), act=None,
#             padding='SAME', W_init=w_init, b_init=None)
#     net = BatchNormLayer(net, #act=lambda x: tl.act.lrelu(x, 0.2),
#             is_train=is_train, gamma_init=gamma_init)
#     net_h4 = ElementwiseLayer(layer=[net_h3, net], combine_fn=tf.add)
#     net_h4.outputs = tl.act.lrelu(net_h4.outputs, 0.2)
#
#     if t_txt is not None:
#         net_txt = InputLayer(t_txt)
#         net_txt = DenseLayer(net_txt, n_units=t_dim,
#                act=lambda x: tl.act.lrelu(x, 0.2),
#                W_init=w_init)
#         net_txt = ExpandDimsLayer(net_txt, 1)
#         net_txt = ExpandDimsLayer(net_txt, 1)
#         net_txt = TileLayer(net_txt, [1, 4, 4, 1])
#         net_h4_concat = ConcatLayer([net_h4, net_txt], concat_dim=3)
#         # 243 (ndf*8 + 128 or 256) x 4 x 4
#         net_h4 = Conv2d(net_h4_concat, df_dim*8, (1, 1), (1, 1),
#                 padding='VALID', W_init=w_init, b_init=None)
#         net_h4 = BatchNormLayer(net_h4, act=lambda x: tl.act.lrelu(x, 0.2),
#                 is_train=is_train, gamma_init=gamma_init)
#
#     net_ho = Conv2d(net_h4, 1, (s16, s16), (s16, s16), padding='VALID', W_init=w_init)
#     # 1 x 1 x 1
#     # net_ho = FlattenLayer(net_h4, name='d_ho/flatten')
#     logits = net_ho.outputs
#     net_ho.outputs = tf.nn.sigmoid(net_ho.outputs)
#     return net_ho.outputs
def conv_cond_concat(x, y):
  """Concatenate conditioning vector on feature map axis."""
  x_shapes = x.get_shape()
  y_shapes = y.get_shape()
  return tf.concat([
    x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

initializer = tf.truncated_normal_initializer(stddev=0.02)
def discriminator(x, y):

    yb = tf.reshape(y, [-1, 1, 1, t_dim])
    h = tf.reshape(x, [-1, 64, 64, 3])

    # h = conv_cond_concat(h, yb)
    h = tf.contrib.layers.conv2d(h, 64, 5, stride=2, padding='SAME', activation_fn=None, weights_initializer=initializer)
    h = tf.contrib.layers.batch_norm(h, activation_fn=tf.nn.relu)

    # h = conv_cond_concat(h, yb)
    h = tf.contrib.layers.conv2d(h, 64*2, 5, stride=2, padding='SAME', activation_fn=None, weights_initializer=initializer)
    h = tf.contrib.layers.batch_norm(h, activation_fn=tf.nn.relu)

    # h = conv_cond_concat(h, yb)
    h = tf.contrib.layers.conv2d(h, 64*4, 5, stride=2, padding='SAME', activation_fn=None, weights_initializer=initializer)
    h = tf.contrib.layers.batch_norm(h, activation_fn=tf.nn.relu)

    h = conv_cond_concat(h, yb)
    h = tf.contrib.layers.conv2d(h, 64*8, 5, stride=2, padding='SAME', activation_fn=tf.nn.relu, weights_initializer=initializer)
    h = tf.contrib.layers.batch_norm(h, activation_fn=tf.nn.relu)
    h = tf.contrib.layers.flatten(h)
    return tf.contrib.layers.fully_connected(h, 1, activation_fn=tf.sigmoid)

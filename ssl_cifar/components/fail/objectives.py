'''
objectives
'''
import numpy as np
import theano.tensor as T
import theano
import lasagne
from theano.tensor.extra_ops import to_one_hot

def balance_constraint_d(predictions, num_classes, norm_type=2, epsilon=1e-6):
    predictions = T.clip(predictions, epsilon, 1-epsilon)
    predictions = predictions.reshape((-1, num_classes))
    res = predictions.mean(axis=0)
    res -= res.mean()
    return res.norm(norm_type)

def categorical_crossentropy(predictions, targets, epsilon=1e-6):
    # avoid overflow
    predictions = T.clip(predictions, epsilon, 1-epsilon)
    # check shape of targets
    num_cls = predictions.shape[1]
    if targets.ndim == predictions.ndim - 1:
        targets = theano.tensor.extra_ops.to_one_hot(targets, num_cls)
    elif targets.ndim != predictions.ndim:
        raise TypeError('rank mismatch between targets and predictions')
    return lasagne.objectives.categorical_crossentropy(predictions, targets).mean()

def entropy(predictions):
    return categorical_crossentropy(predictions, predictions)

def negative_entropy_of_mean(predictions):
    return -entropy(predictions.mean(axis=0, keepdims=True))

def categorical_crossentropy_of_mean(predictions):
    num_cls = predictions.shape[1]
    uniform_targets = T.ones((1, num_cls)) / num_cls
    return categorical_crossentropy(predictions.mean(axis=0, keepdims=True), uniform_targets)

def categorical_crossentropy_ssl_alternative(predictions, targets, num_labelled, weight_decay, alpha_labeled=1., alpha_unlabeled=.3, alpha_average=1e-3, alpha_decay=1e-4):
    ce_loss = categorical_crossentropy(predictions[:num_labelled], targets)
    en_loss = entropy(predictions[num_labelled:])
    av_loss = negative_entropy_of_mean(predictions[num_labelled:])
    return alpha_labeled*ce_loss + alpha_unlabeled*en_loss + alpha_average*av_loss + alpha_decay*weight_decay

def categorical_crossentropy_ssl(predictions, targets, num_labelled, weight_decay, alpha_labeled=1., alpha_unlabeled=.3, alpha_average=1e-3, alpha_decay=1e-4):
    ce_loss = categorical_crossentropy(predictions[:num_labelled], targets)
    en_loss = entropy(predictions[num_labelled:])
    av_loss = categorical_crossentropy_of_mean(predictions[num_labelled:])
    return alpha_labeled*ce_loss + alpha_unlabeled*en_loss + alpha_average*av_loss + alpha_decay*weight_decay

def categorical_crossentropy_ssl_separated(predictions_l, targets, predictions_u, weight_decay, alpha_labeled=1., alpha_unlabeled=.3, alpha_average=1e-3, alpha_decay=1e-4):
    ce_loss = categorical_crossentropy(predictions_l, targets)
    en_loss = entropy(predictions_u)
    av_loss = categorical_crossentropy_of_mean(predictions_u)
    return alpha_labeled*ce_loss + alpha_unlabeled*en_loss + alpha_average*av_loss + alpha_decay*weight_decay

def maximum_mean_discripancy(sample, data, sigma=[2,5,10,20,40,80]):
    x = T.concatenate([sample, data], axis=0)
    xx = T.dot(x, x.T)
    x2 = T.sum(x*x, axis=1, keepdims=True)
    exponent = xx - .5*x2 - .5*x2.T
    s_samples = T.ones([sample.shape[0], 1])*1./sample.shape[0]
    s_data = -T.ones([data.shape[0], 1])*1./data.shape[0]
    s_all = T.concatenate([s_samples, s_data], axis=0)
    s_mat = T.dot(s_all, s_all.T)
    mmd_loss = 0.
    for s in sigma:
        kernel_val = T.exp(1./s * exponent)
        mmd_loss += T.sum(s_mat*kernel_val)
    return T.sqrt(mmd_loss)

def conditional_maximum_mean_discripancy(sample, data, label, num_classes, sigma=[2,5,10,20,40,80]):
    cmmd_loss = 0.
    for c in xrange(num_classes):
        index = T.nonzero(T.switch(T.eq(label, c), T.ones_like(label), T.zeros_like(label)))
        cmmd_loss += maximum_mean_discripancy(sample[index], data[index], sigma)
    return cmmd_loss

def feature_matching(f_sample, f_data, norm='l2'):
    if norm == 'l2':
        return T.mean(T.square(T.mean(f_sample,axis=0)-T.mean(f_data,axis=0)))
    elif norm == 'l1':
        return T.mean(abs(T.mean(f_sample,axis=0)-T.mean(f_data,axis=0)))
    else:
        raise NotImplementedError

def conditional_feature_matching(f_sample, f_data, label, num_classes, norm='l2'):
    cfm_loss = 0.
    for c in xrange(num_classes):
        index = T.nonzero(T.switch(T.eq(label, c), T.ones_like(label), T.zeros_like(label)))
        cfm_loss += feature_matching(f_sample[index], f_data[index], norm)
    return cfm_loss

def entropy_of_unlabeled_d(p_vals, num_classes, epsilon=1e-6):
    p_vals = T.clip(p_vals, epsilon, 1-epsilon)
    p_vals = p_vals.reshape((-1, num_classes))
    z = p_vals.sum(axis=1, keepdims=True)
    p_vals = p_vals / z
    return entropy(p_vals)

def margin_of_unlabeled_d(p_vals, num_classes, epsilon=1e-6):
    p_vals = T.clip(p_vals, epsilon, 1-epsilon)
    p_vals = p_vals.reshape((-1, num_classes))
    p_max = p_vals.max(axis=1)
    p_res = -(p_max - p_vals.mean(axis=1))
    return p_res.mean()

bce = lasagne.objectives.binary_crossentropy
def bce_of_unlabeled_d(p_vals, num_classes, negative_samples=False, epsilon=1e-6):
    p_vals = T.clip(p_vals, epsilon, 1-epsilon)
    p_vals = p_vals.reshape((-1, num_classes))
    if not negative_samples:
        p_max = p_vals.max(axis=1)
        return bce(p_max, T.ones(p_max.shape)).mean() 
    else:
        p_max_index = p_vals.argmax(axis=1)
        p_max_index = to_one_hot(p_max_index, num_classes)
        return bce(p_vals, p_max_index).mean()

def estimates_of_unlabeled_d(p_vals, num_classes, epsilon=1e-6):
    p_vals = T.clip(p_vals, epsilon, 1-epsilon)
    p_vals = p_vals.reshape((-1, num_classes))
    z = p_vals.sum(axis=1, keepdims=True)
    p_vals_norm = p_vals / z
    return -((p_vals_norm*T.log(p_vals)).sum(axis=1)).mean()

def estimates_by_c_of_unlabeled_d(p_vals, p_vals_norm_c, num_classes, epsilon=1e-6):
    p_vals = T.clip(p_vals, epsilon, 1-epsilon)
    p_vals = p_vals.reshape((-1, num_classes))
    return -((p_vals_norm_c*T.log(p_vals)).sum(axis=1)).mean()
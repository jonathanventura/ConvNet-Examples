from keras import backend as K
import numpy as np
import cv2

def balanced_binary_crossentropy(y_true, y_pred):
    """Binary crossentropy loss function with automatic class balancing"""
    y_true_f = K.cast(y_true,K.floatx())
    num_pos = K.sum(y_true_f)+1
    num_neg = K.sum(1-y_true_f)+1
    #weights = 1/(2*num_pos)*y_true + 1/(2*num_neg)*(1-y_true)
    weights = (num_neg)/(num_pos)*y_true_f + (1-y_true_f)
    score_array = K.binary_crossentropy(y_pred,y_true_f)
    score_array *= weights
    score_array /= K.mean(K.cast(K.not_equal(weights, 0), K.floatx()))
    return K.mean(score_array)

def pad_images(imgs,pad):
    """Pad images with reflection padding"""
    imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1]+pad*2, imgs.shape[2]+pad*2, imgs.shape[3]), dtype=imgs.dtype)
    for i in xrange(imgs.shape[0]):
        imgs_p[i,:,:,:] = cv2.copyMakeBorder(imgs[i,:,:,:],pad,pad,pad,pad,cv2.BORDER_REFLECT)
    return imgs_p

def remove_mean(imgs):
    """Subtract per-channel mean from each image"""
    for i in xrange(imgs.shape[0]):
        for j in xrange(imgs.shape[3]):
            imgs[i,:,:,j] -= np.mean(imgs[i,:,:,j])
    return imgs


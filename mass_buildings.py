from __future__ import absolute_import
from keras.utils.data_utils import get_file
from keras import backend as K
import numpy as np
import os
import cv2


def load_data():
    train_nbs=(22678915,22678930,22678945,22678960,22678975,22678990,22679005,22679020,22679035,22679050,22828915,22828945,22828960,22828975,22829005,22829020,22829035,22978870,22978885,22978900,22978915,22978930,22978960,22978975,22978990,22979005,22979020,22979035,22979050,22979065,23128870,23128885,23128900,23128915,23128930,23128945,23128960,23128975,23128990,23129005,23129020,23129035,23129050,23129065,23129125,23129140,23129155,23129170,23278885,23278900,23278915,23278930,23278945,23278960,23278975,23278990,23279005,23279020,23279035,23279050,23279080,23279095,23279140,23279155,23279170,23428900,23428915,23428930,23428945,23428960,23428975,23428990,23429005,23429035,23429050,23429065,23429095,23429125,23429140,23429170,23578915,23578930,23578945,23578975,23578990,23579020,23579035,23579065,23579080,23579095,23579110,23579125,23579140,23728840,23728945,23728960,23728975,23728990,23729005,23729020,23729050,23729065,23729080,23729095,23729110,23878915,23878930,23878945,23878975,23878990,23879020,23879035,23879050,23879065,23879095,23879110,24029035,24029050,24029065,24029080,24029110,24179020,24179035,24179050,24179080,24328840,24328855,24328870,24329020,24329035,24329095,24478840,24478855,24478870,24478885,24478900,24479005)
    test_nbs=(22828930,22828990,22829050,23429020,23429080,23578960,23579005,23729035,23879080,24179065)

    dirname = "mass-buildings-py"
    baseorigin = "http://www.cs.toronto.edu/~vmnih/data/mass_buildings"

    nb_train_samples = 10 #len(train_nbs)

    X_train = np.zeros((nb_train_samples, 1500, 1500, 3), dtype="uint8")
    y_train = np.zeros((nb_train_samples, 1500, 1500, 1), dtype="uint8")

    for i in xrange(nb_train_samples):
        nb = train_nbs[i]
        path = get_file('%d_15.tiff'%nb, origin='%s/train/sat/%d_15.tiff'%(baseorigin,nb))
        X_train[i,:,:,:] = cv2.imread(path)
        path = get_file('%d_15.tif'%nb, origin='%s/train/map/%d_15.tif'%(baseorigin,nb))
        y_train[i,:,:,0] = cv2.imread(path)[:,:,0] 

    nb_test_samples = len(test_nbs)

    X_test = np.zeros((nb_test_samples, 1500, 1500, 3), dtype="uint8")
    y_test = np.zeros((nb_test_samples, 1500, 1500, 1), dtype="uint8")

    for i in xrange(nb_test_samples):
        nb = test_nbs[i]
        path = get_file('%d_15.tiff'%nb, origin='%s/test/sat/%d_15.tiff'%(baseorigin,nb))
        X_test[i,:,:,:] = cv2.imread(path)
        path = get_file('%d_15.tif'%nb, origin='%s/test/map/%d_15.tif'%(baseorigin,nb))
        y_test[i,:,:,0] = cv2.imread(path)[:,:,0] 

    if K.image_dim_ordering() == 'theano':
        X_train = X_train.transpose(0, 3, 1, 2)
        y_train = X_train.transpose(0, 3, 1, 2)
        X_test = X_test.transpose(0, 3, 1, 2)
        y_test = X_test.transpose(0, 3, 1, 2)

    return (X_train, y_train), (X_test, y_test)

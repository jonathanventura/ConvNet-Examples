from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Activation, Reshape, Permute, Deconvolution2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, SGD, Adagrad
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import os
import sys
import mass_buildings
from common import *

batch_size = 1
model_path = 'model.hdf5'

def make_convnet(image_rows,image_cols):
    inputs = Input((image_rows, image_cols, 3))

    # 3x3 convolution layer
    x = Convolution2D(32, 3, 3, border_mode='valid', init='he_normal')(inputs)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    # 3x3 convolution layer
    x = Convolution2D(32, 3, 3, border_mode='valid', init='he_normal')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    # 3x3 convolution layer
    x = Convolution2D(32, 3, 3, border_mode='valid', init='he_normal')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    # 1x1 convolution layer
    # This is essentially a per-pixel fully connected layer,
    # which makes this a "fully convolutional network"
    x = Convolution2D(1, 1, 1, init='he_normal')(x)
    
    # Sigmoid activation for binary classification
    x = Activation('sigmoid')(x)

    model = Model(input=inputs, output=x)

    opt = Adagrad()
    model.compile(optimizer=opt, loss=balanced_binary_crossentropy, metrics=['binary_accuracy','precision','recall'])

    return model

def train_model():
    """Train the ConvNet model"""
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = mass_buildings.load_data()
    
    X_train = pad_images(X_train,3)
    X_val = pad_images(X_val,3)
    X_test = pad_images(X_test,3)

    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    X_test = X_test.astype('float32')

    X_train = remove_mean(X_train)
    X_val = remove_mean(X_val)
    X_test = remove_mean(X_test)

    if not os.path.isfile(model_path):
        model = make_convnet(X_train.shape[1],X_train.shape[2])
    
        model_checkpoint = ModelCheckpoint(model_path, monitor='loss', save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=10, verbose=0, mode='auto')
        model.fit(X_train, y_train, 
                  validation_data=(X_val,y_val),
                  batch_size=batch_size, nb_epoch=10, verbose=1, shuffle=True,
                  callbacks=[model_checkpoint,early_stopping])
        model.save_weights(model_path)

    #model = make_convnet(imgs_test.shape[1],imgs_test.shape[2])
    #model.load_weights(model_path)
    
if __name__ == '__main__':
    train_model()

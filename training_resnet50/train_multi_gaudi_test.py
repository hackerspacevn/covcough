### TO run all 8 cores: mpirun --allow-run-as-root -np 8 python3 train-multi.py

import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras.layers import concatenate, Activation, Dense, Dropout, Conv2D, Flatten, MaxPool2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling1D, AveragePooling2D, Input, Add, BatchNormalization, ZeroPadding2D
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras import layers
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot
from tensorflow.keras.applications import ResNet50

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import ModelCheckpoint
import librosa
import librosa.display
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

## Loading habana frameworks bridge
from habana_frameworks.tensorflow import load_habana_module
load_habana_module()

import horovod.tensorflow.keras as hvd
#Initialization of Horovod. 
hvd.init()

# Ensure only 1 process downloads the data on each node
if hvd.local_rank() == 0:
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() 
    hvd.broadcast(0, 0)
else:
	hvd.broadcast(0, 0)
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()



x_train=np.load('./trainingdata/melspecs_train_64x64_centric_2022(4-samplerates)-003.npy')
y_train=np.load('./trainingdata/y_result_train_64x64_centric_2022(4-samplerates).npy')

x_test=np.load('./trainingdata/melspecs_test_64x64_centric_2022(4-samplerates)-002.npy')
y_test=np.load('./trainingdata/y_result_test_64x64_centric_2022(4-samplerates).npy')

# X_valid=np.load('./trainingdata/melspecs_valid_64x64_2022.npy')
# y_valid=np.load('./trainingdata/y_result_valid_64x64_2022.npy')

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


def identity_block(X, f, filters, stage, block):
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. We'll need this later to add back to the main path. 
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def convolutional_block(X, f, filters, stage, block, s = 2):
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    
    ##### SHORTCUT PATH ####
    X_shortcut = Conv2D(F3, (1, 1), strides = (s,s), name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X

def ResNet50(input_shape = (64, 64, 1), classes = 2):   
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL.
    X = AveragePooling2D((2, 2), name='avg_pool')(X)

    # output layer
    X = Flatten()(X)
    X = BatchNormalization()(X)

    X = Dense(4096, activation='relu')(X)
    X = BatchNormalization()(X)
    X = Dropout(0.4)(X)
    
    X = Dense(1024, activation='relu')(X)
    X = BatchNormalization()(X)
    X = Dropout(0.4)(X)

    X = Dense(512, activation='relu')(X)
    X = BatchNormalization()(X)
    X = Dropout(0.4)(X)
    

    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model

def identity_block(X, f, filters, stage, block):
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. We'll need this later to add back to the main path. 
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    X = Dropout(0.4)(X)

    # Second component of main path
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)
    X = Dropout(0.4)(X)

    # Third component of main path
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)
    X = Dropout(0.4)(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def convolutional_block(X, f, filters, stage, block, s = 2):
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    X = Dropout(0.4)(X)

    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)
    X = Dropout(0.4)(X)

    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
    X = Dropout(0.4)(X)

    
    ##### SHORTCUT PATH ####
    X_shortcut = Conv2D(F3, (1, 1), strides = (s,s), name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X


num_pics_per_rank = x_train.shape[0] // hvd.size()
pic_begin = num_pics_per_rank * hvd.rank()
pic_end = pic_begin + num_pics_per_rank
x_train = x_train[pic_begin:pic_end, ]
y_train = y_train[pic_begin:pic_end, ]

x_train, x_test = x_train / 255.0, x_test / 255.0

model2 = ResNet50(input_shape = (64,64,1), classes = 2)
#Model Summary
model2.summary()

for layer in model2.layers:
    layer.trainable = True

# Compiling the model
#opt = Adam(lr=0.001)
optimizer = Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
optimizer =hvd.DistributedOptimizer(optimizer)

callbacks = [
    hvd.callbacks.BestModelCheckpoint(filepath='./Early_CoughCovid_ResNet50_checkpoint.hdf5',monitor='val_loss',verbose=1,save_best_only=True),
]

model2.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer=optimizer)
num_epoch=1
num_batch_size=32
# checkpointer=ModelCheckpoint(filepath='./Early_CoughCovid_ResNet50_checkpoint.hdf5',monitor='val_loss',verbose=1,save_best_only=True)

# historyCNN2=model2.fit(x_train,y_train,batch_size=num_batch_size,epochs=num_epoch,validation_data=(x_test,y_test),callbacks=[checkpointer])
historyCNN2=model2.fit(x_train,y_train,batch_size=num_batch_size,epochs=num_epoch,validation_data=(x_test,y_test),callbacks=callbacks)

model2.save('./Final_CoughCovid_ResNet50(64x64)15-02-2022-size32.hdf5')



plt.figure(figsize=(10,5))
plt.ylim([0,2])
plt.plot(historyCNN2.history['loss'])
plt.plot(historyCNN2.history['val_loss'])
plt.plot(historyCNN2.history['accuracy'])
plt.plot(historyCNN2.history['val_accuracy'])
plt.title('ResNet50 64x64 (19-01-2022)')
plt.ylabel('value')
plt.xlabel('epoch')
plt.legend(['loss','val_loss','accuracy','val_accuracy'],bbox_to_anchor=(1.2, 1.0), loc='upper right')
plt.savefig('training_result.png')
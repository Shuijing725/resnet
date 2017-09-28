from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json

from keras.models import Sequential, Model
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Add
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import add
from keras.regularizers import l2

import numpy as np

from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras import optimizers


def build_resnet18(input_size):
	# input block
	inputs = Input(shape = input_size)_
	conv1_cov = Convolution2D(filters = 64, kernel_size = (7, 7), strides = (2, 2), \
		padding="valid", kernel_intializer = 'he_normal', kernel_regularizer = l2(1e-4))(inputs)
	conv1_pad = ZeroPadding2D(padding = (3, 3))(conv1_cov)
	conv1_batch = BatchNormalization(axis = 3, scale = True)(conv1_pad)
	conv1_relu = Activation('relu')(conv1_batch)

	# first pooling
	pool1 = MaxPooling2D(pool_size = (3, 3), strides = 2)(conv1_relu)

	# first residual block: implemented in the proposed version, not original
	# bn -> relu -> convolution -> bn -> relu -> convolution
	conv2_bn1 = BatchNormalization(axis = 3, scale = True)(pool1)
	conv2_relu1 = Activation('relu')(conv2_bn1)
	conv2_weight1 = Convolution2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), \
		padding="valid", kernel_intializer = 'he_normal', kernel_regularizer = l2(1e-4))(conv2_relu1)
	
	conv2_bn2 = BatchNormalization(axis = 3, scale = True)(conv2_weight1)
	conv2_relu2 = Activation('relu')(conv2_bn2)
	conv2_weight2 = Convolution2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), \
		padding="valid", kernel_intializer = 'he_normal', kernel_regularizer = l2(1e-4))(conv2_relu2)

	add2 = Add([conv2_weight2, pool1])
	relu_add2 = Activation('relu')(add2)

	# second block
	conv3_bn1 = BatchNormalization(axis = 3, scale = True)(relu_add2)
	conv3_relu1 = Activation('relu')(conv3_bn1)
	conv3_weight1 = Convolution2D(filters = 128, kernel_size = (3, 3), strides = (1, 1), \
		padding="valid", kernel_intializer = 'he_normal', kernel_regularizer = l2(1e-4))(conv3_relu1)
	
	conv3_bn2 = BatchNormalization(axis = 3, scale = True)(conv3_weight1)
	conv3_relu2 = Activation('relu')(conv3_bn2)
	conv3_weight2 = Convolution2D(filters = 128, kernel_size = (3, 3), strides = (1, 1), \
		padding="valid", kernel_intializer = 'he_normal', kernel_regularizer = l2(1e-4))(conv3_relu2)

	add3 = Add([conv2_weight2, relu_add2])
	relu_add3= Activation('relu')(add3)		
    
    # third block
	conv4_bn1 = BatchNormalization(axis = 3, scale = True)(relu_add3)
	conv4_relu1 = Activation('relu')(conv4_bn1)
	conv4_weight1 = Convolution2D(filters = 256, kernel_size = (3, 3), strides = (1, 1), \
		padding="valid", kernel_intializer = 'he_normal', kernel_regularizer = l2(1e-4))(conv4_relu1)
	
	conv4_bn2 = BatchNormalization(axis = 3, scale = True)(conv4_weight1)
	conv4_relu2 = Activation('relu')(conv4_bn2)
	conv4_weight2 = Convolution2D(filters = 256, kernel_size = (3, 3), strides = (1, 1), \
		padding="valid", kernel_intializer = 'he_normal', kernel_regularizer = l2(1e-4))(conv4_relu2)

	add4 = Add([conv4_weight2, relu_add3])
	relu_add4 = Activation('relu')(add4)	        

	# last block
	conv5_bn1 = BatchNormalization(axis = 3, scale = True)(relu_add4)
	conv5_relu1 = Activation('relu')(conv5_bn1)
	conv5_weight1 = Convolution2D(filters = 512, kernel_size = (3, 3), strides = (1, 1), \
		padding="valid", kernel_intializer = 'he_normal', kernel_regularizer = l2(1e-4))(conv5_relu1)
	
	conv5_bn2 = BatchNormalization(axis = 3, scale = True)(conv5_weight1)
	conv5_relu2 = Activation('relu')(conv5_bn2)
	conv5_weight2 = Convolution2D(filters = 512, kernel_size = (3, 3), strides = (1, 1), \
		padding="valid", kernel_intializer = 'he_normal', kernel_regularizer = l2(1e-4))(conv5_relu2)
	add5 = Add([conv5_weight2, relu_add4])
	relu_add5 = Activation('relu')(add5)	

	# end 
	pool5 = AveragePooling2D(pool_size = (7, 7), strides = 1)(relu_add5)
	flatten = Flatten()(pool5)
	fc = Dense(1000, kernel_intializer = 'he_normal', kernel_regularizer = l2(1e-4), \
		activation = 'softmax')(flatten)
	output = Dense(1)(fc)

	return Model(inputs, output)


"""
parameters
"""
# dimensions of images.
img_width, img_height = 128, 128
img_channels = 3
# directory
train_data_dir = '/home/shuijing/cancer-prediction/deep_learning/datasets/first_dataset/train'
validation_data_dir = '/home/shuijing/cancer-prediction/deep_learning/datasets/first_dataset/test'
# parameters
nb_train_samples = 6000
nb_validation_samples = 2000
nb_epoch = 50
nb_batch = 32
nb_classes = 2

# callbacks for stable loss
lr_reducer = ReduceLROnPlateau (factor=0.1, cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.0001, patience=10)
csv_logger = CSVLogger('resnet18.csv')

# build model
model = build_resnet18((img_channels, img_width, img_height))
# myoptimizer = optimizers.SGD(lr=0.1, momentum = 0.9, nesterov = True)
myoptimizer = optimizers.RMSprop()
# myoptimizer = optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='binary_crossentropy',
              optimizer=myoptimizer,
              metrics=['accuracy'])

"""
load data
"""
# training data
train_datagen = ImageDataGenerator(rescale=1./255)
# 
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=nb_batch,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=nb_batch,
        class_mode='binary')


model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        # validation_split = 0.25,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples,
        callbacks=[lr_reducer, early_stopper, csv_logger])

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("weights.h5")
print("Saved model to disk")

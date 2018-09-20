import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import io
import sys
sys.path.append("./networks")
import numpy as np
import tensorflow as tf
keras=tf.contrib.keras
l2=keras.regularizers.l2
K=tf.contrib.keras.backend
import inputs as data
from res3d_cgru_mobilenet import res3d_cgru_mobilenet
from datagen import isoTestImageGenerator,jesterTestImageGenerator
from datetime import datetime

# Used ConvLSTM Type
SEPARABLE = 0
GROUP = 1
SHUFFLE = 2
GATED = 3

# Modality
RGB = 0
Depth = 1
Flow = 2

# Dataset
JESTER = 0
ISOGD = 1

cfg_type = GATED
cfg_modality = RGB
cfg_dataset = ISOGD

if cfg_dataset==JESTER:
  seq_len = 16
  batch_size = 16
  num_classes = 27
  testing_datalist = './dataset_splits/Jester/valid_rgb_list.txt'
elif cfg_dataset==ISOGD:
  seq_len = 32
  batch_size = 8
  num_classes = 249
  testing_datalist = './dataset_splits/IsoGD/valid_rgb_list.txt'

weight_decay = 0.00005
model_prefix = '/raid/gmzhu/tensorflow/ConvLSTMForGR/models/'
  
inputs = keras.layers.Input(shape=(seq_len, 112, 112, 3),
                            batch_shape=(batch_size, seq_len, 112, 112, 3))
feature = res3d_cgru_mobilenet(inputs, seq_len, weight_decay, cfg_type)
flatten = keras.layers.Flatten(name='Flatten')(feature)
classes = keras.layers.Dense(num_classes, activation='linear', kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), name='Classes')(flatten)
outputs = keras.layers.Activation('softmax', name='Output')(classes)
model = keras.models.Model(inputs=inputs, outputs=outputs)
optimizer = keras.optimizers.SGD(lr=0.001, decay=0, momentum=0.9, nesterov=False)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

if cfg_dataset==JESTER:
  if cfg_type==SEPARABLE:
    pretrained_model = '%s/jester_rgb_separablecgru_weights.h5'%model_prefix
  elif cfg_type==SHUFFLE:
    pretrained_model = '%s/jester_rgb_shufflecgru_weights.h5'%model_prefix
  elif cfg_type==GATED:
    pretrained_model = '%s/jester_rgb_gatedcgru_weights.h5'%model_prefix
elif cfg_dataset==ISOGD:
  if cfg_type==SEPARABLE:
    pretrained_model = '%s/isogr_rgb_separablecgru_weights.h5'%model_prefix
  elif cfg_type==SHUFFLE:
    pretrained_model = '%s/isogr_rgb_shufflecgru_weights.h5'%model_prefix
  elif cfg_type==GATED:
    pretrained_model = '%s/isogr_rgb_gatedcgru_weights.h5'%model_prefix
print 'Loading pretrained model from %s' % pretrained_model
model.load_weights(pretrained_model, by_name=False)
for i in range(len(model.trainable_weights)):
  print model.trainable_weights[i]

_,test_labels = data.load_iso_video_list(testing_datalist)
test_steps = len(test_labels)/batch_size
if cfg_dataset==JESTER:
  print model.evaluate_generator(jesterTestImageGenerator(testing_datalist, batch_size, seq_len, num_classes, cfg_modality),
                         steps=test_steps,
                         )
elif cfg_dataset==ISOGD:
  print model.evaluate_generator(isoTestImageGenerator(testing_datalist, batch_size, seq_len, num_classes, cfg_modality),
                         steps=test_steps,
                         )

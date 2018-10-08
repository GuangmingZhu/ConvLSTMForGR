import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import io
import sys
sys.path.append("./networks")
import numpy as np
import tensorflow as tf
keras=tf.contrib.keras
l2=keras.regularizers.l2
K=tf.contrib.keras.backend
import inputs as data
from res3d_clstm_mobilenet import res3d_clstm_mobilenet
from datagen import jesterTestImageGenerator
from datagen import isoTestImageGenerator
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
cfg_pyramid = False

if cfg_modality==RGB:
  str_modality = 'rgb'
elif cfg_modality==Depth:
  str_modality = 'depth'
elif cfg_modality==Flow:
  str_modality = 'flow'

if cfg_dataset==JESTER:
  seq_len = 16
  batch_size = 16
  num_classes = 27
  testing_datalist = './dataset_splits/Jester/valid_%s_list.txt'%str_modality
elif cfg_dataset==ISOGD:
  seq_len = 32
  batch_size = 8
  num_classes = 249
  testing_datalist = './dataset_splits/IsoGD/valid_%s_list.txt'%str_modality

weight_decay = 0.00005
model_prefix = '/raid/gmzhu/tensorflow/ConvLSTMForGR/models/'
  
if cfg_pyramid==True:
  inputs = keras.layers.Input(shape=(seq_len, 112, 112, 3), batch_shape=(batch_size*4, seq_len, 112, 112, 3))
else:
  inputs = keras.layers.Input(shape=(seq_len, 112, 112, 3), batch_shape=(batch_size, seq_len, 112, 112, 3))
feature = res3d_clstm_mobilenet(inputs, seq_len, weight_decay, cfg_type)
# Pyramid Pooling
if cfg_pyramid==True:
  feature = keras.layers.Reshape((4,1024), name='Pyramid_reshape')(feature)
  feature = keras.layers.AveragePooling1D(pool_size=4, strides=4, name='Pyramid_Pooling')(feature)

flatten = keras.layers.Flatten(name='Flatten')(feature)
classes = keras.layers.Dense(num_classes, activation='linear', kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), name='Classes')(flatten)
outputs = keras.layers.Activation('softmax', name='Output')(classes)
model = keras.models.Model(inputs=inputs, outputs=outputs)
optimizer = keras.optimizers.SGD(lr=0.001, decay=0, momentum=0.9, nesterov=False)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

if cfg_dataset==JESTER:
  if cfg_type==SEPARABLE:
    pretrained_model = '%s/jester_%s_separableclstm_weights.h5'%(model_prefix,str_modality)
  elif cfg_type==GROUP:
    pretrained_model = '%s/jester_%s_groupclstm_weights.h5'%(model_prefix,str_modality)
  elif cfg_type==SHUFFLE:
    pretrained_model = '%s/jester_%s_shuffleclstm_weights.h5'%(model_prefix,str_modality)
  elif cfg_type==GATED:
    pretrained_model = '%s/jester_%s_gatedclstm_weights.h5'%(model_prefix,str_modality)
elif cfg_dataset==ISOGD:
  if cfg_type==SEPARABLE:
    pretrained_model = '%s/isogr_%s_separableclstm_weights.h5'%(model_prefix,str_modality)
  elif cfg_type==GROUP:
    pretrained_model = '%s/isogr_%s_groupclstm_weights.h5'%(model_prefix,str_modality)
  elif cfg_type==SHUFFLE:
    pretrained_model = '%s/isogr_%s_shuffleclstm_weights.h5'%(model_prefix,str_modality)
  elif cfg_type==GATED:
    pretrained_model = '%s/isogr_%s_gatedclstm_weights.h5'%(model_prefix,str_modality)
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

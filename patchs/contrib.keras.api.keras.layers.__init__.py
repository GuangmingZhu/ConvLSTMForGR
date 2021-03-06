# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Keras layers API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Generic layers.
# pylint: disable=g-bad-import-order
from tensorflow.contrib.keras.python.keras.engine import Input
from tensorflow.contrib.keras.python.keras.engine import InputLayer
from tensorflow.contrib.keras.python.keras.engine import InputSpec
from tensorflow.contrib.keras.python.keras.engine import Layer

# Advanced activations.
from tensorflow.contrib.keras.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.contrib.keras.python.keras.layers.advanced_activations import PReLU
from tensorflow.contrib.keras.python.keras.layers.advanced_activations import ELU
from tensorflow.contrib.keras.python.keras.layers.advanced_activations import ThresholdedReLU

# Convolution layers.
from tensorflow.contrib.keras.python.keras.layers.convolutional import Conv1D
from tensorflow.contrib.keras.python.keras.layers.convolutional import Conv2D
from tensorflow.contrib.keras.python.keras.layers.convolutional import Conv3D
from tensorflow.contrib.keras.python.keras.layers.convolutional import Conv2DTranspose
from tensorflow.contrib.keras.python.keras.layers.convolutional import SeparableConv2D

# Convolution layer aliases.
from tensorflow.contrib.keras.python.keras.layers.convolutional import Convolution1D
from tensorflow.contrib.keras.python.keras.layers.convolutional import Convolution2D
from tensorflow.contrib.keras.python.keras.layers.convolutional import Convolution3D
from tensorflow.contrib.keras.python.keras.layers.convolutional import Convolution2DTranspose
from tensorflow.contrib.keras.python.keras.layers.convolutional import SeparableConvolution2D

# Image processing layers.
from tensorflow.contrib.keras.python.keras.layers.convolutional import UpSampling1D
from tensorflow.contrib.keras.python.keras.layers.convolutional import UpSampling2D
from tensorflow.contrib.keras.python.keras.layers.convolutional import UpSampling3D
from tensorflow.contrib.keras.python.keras.layers.convolutional import ZeroPadding1D
from tensorflow.contrib.keras.python.keras.layers.convolutional import ZeroPadding2D
from tensorflow.contrib.keras.python.keras.layers.convolutional import ZeroPadding3D
from tensorflow.contrib.keras.python.keras.layers.convolutional import Cropping1D
from tensorflow.contrib.keras.python.keras.layers.convolutional import Cropping2D
from tensorflow.contrib.keras.python.keras.layers.convolutional import Cropping3D

# Convolutional-recurrent layers.
from tensorflow.contrib.keras.python.keras.layers.convolutional_recurrent import ConvLSTM2D
from tensorflow.contrib.keras.python.keras.layers.convolutional_recurrent import GatedConvLSTM2D
from tensorflow.contrib.keras.python.keras.layers.convolutional_recurrent import GatedConvGRU2D
from tensorflow.contrib.keras.python.keras.layers.convolutional_recurrent import SeparableConvLSTM2D
from tensorflow.contrib.keras.python.keras.layers.convolutional_recurrent import SeparableConvGRU2D
from tensorflow.contrib.keras.python.keras.layers.convolutional_recurrent import GroupConvLSTM2D
from tensorflow.contrib.keras.python.keras.layers.convolutional_recurrent import ShuffleConvLSTM2D
from tensorflow.contrib.keras.python.keras.layers.convolutional_recurrent import ShuffleConvGRU2D
from tensorflow.contrib.keras.python.keras.layers.convolutional_recurrent import AttenXConvLSTM2D
from tensorflow.contrib.keras.python.keras.layers.convolutional_recurrent import AttenIConvLSTM2D
from tensorflow.contrib.keras.python.keras.layers.convolutional_recurrent import AttenOConvLSTM2D

# Core layers.
from tensorflow.contrib.keras.python.keras.layers.core import Masking
from tensorflow.contrib.keras.python.keras.layers.core import Dropout
from tensorflow.contrib.keras.python.keras.layers.core import SpatialDropout1D
from tensorflow.contrib.keras.python.keras.layers.core import SpatialDropout2D
from tensorflow.contrib.keras.python.keras.layers.core import SpatialDropout3D
from tensorflow.contrib.keras.python.keras.layers.core import Activation
from tensorflow.contrib.keras.python.keras.layers.core import Reshape
from tensorflow.contrib.keras.python.keras.layers.core import Permute
from tensorflow.contrib.keras.python.keras.layers.core import Flatten
from tensorflow.contrib.keras.python.keras.layers.core import RepeatVector
from tensorflow.contrib.keras.python.keras.layers.core import Lambda
from tensorflow.contrib.keras.python.keras.layers.core import Dense
from tensorflow.contrib.keras.python.keras.layers.core import ActivityRegularization

# Embedding layers.
from tensorflow.contrib.keras.python.keras.layers.embeddings import Embedding

# Locally-connected layers.
from tensorflow.contrib.keras.python.keras.layers.local import LocallyConnected1D
from tensorflow.contrib.keras.python.keras.layers.local import LocallyConnected2D

# Merge layers.
from tensorflow.contrib.keras.python.keras.layers.merge import Add
from tensorflow.contrib.keras.python.keras.layers.merge import Multiply
from tensorflow.contrib.keras.python.keras.layers.merge import Average
from tensorflow.contrib.keras.python.keras.layers.merge import Maximum
from tensorflow.contrib.keras.python.keras.layers.merge import Concatenate
from tensorflow.contrib.keras.python.keras.layers.merge import Dot
from tensorflow.contrib.keras.python.keras.layers.merge import add
from tensorflow.contrib.keras.python.keras.layers.merge import multiply
from tensorflow.contrib.keras.python.keras.layers.merge import average
from tensorflow.contrib.keras.python.keras.layers.merge import maximum
from tensorflow.contrib.keras.python.keras.layers.merge import concatenate
from tensorflow.contrib.keras.python.keras.layers.merge import dot

# Noise layers.
from tensorflow.contrib.keras.python.keras.layers.noise import GaussianNoise
from tensorflow.contrib.keras.python.keras.layers.noise import GaussianDropout

# Normalization layers.
from tensorflow.contrib.keras.python.keras.layers.normalization import BatchNormalization

# Pooling layers.
from tensorflow.contrib.keras.python.keras.layers.pooling import MaxPooling1D
from tensorflow.contrib.keras.python.keras.layers.pooling import MaxPooling2D
from tensorflow.contrib.keras.python.keras.layers.pooling import MaxPooling3D
from tensorflow.contrib.keras.python.keras.layers.pooling import AveragePooling1D
from tensorflow.contrib.keras.python.keras.layers.pooling import AveragePooling2D
from tensorflow.contrib.keras.python.keras.layers.pooling import AveragePooling3D
from tensorflow.contrib.keras.python.keras.layers.pooling import GlobalAveragePooling1D
from tensorflow.contrib.keras.python.keras.layers.pooling import GlobalAveragePooling2D
from tensorflow.contrib.keras.python.keras.layers.pooling import GlobalAveragePooling3D
from tensorflow.contrib.keras.python.keras.layers.pooling import GlobalMaxPooling1D
from tensorflow.contrib.keras.python.keras.layers.pooling import GlobalMaxPooling2D
from tensorflow.contrib.keras.python.keras.layers.pooling import GlobalMaxPooling3D

# Pooling layer aliases.
from tensorflow.contrib.keras.python.keras.layers.pooling import MaxPool1D
from tensorflow.contrib.keras.python.keras.layers.pooling import MaxPool2D
from tensorflow.contrib.keras.python.keras.layers.pooling import MaxPool3D
from tensorflow.contrib.keras.python.keras.layers.pooling import AvgPool1D
from tensorflow.contrib.keras.python.keras.layers.pooling import AvgPool2D
from tensorflow.contrib.keras.python.keras.layers.pooling import AvgPool3D
from tensorflow.contrib.keras.python.keras.layers.pooling import GlobalAvgPool1D
from tensorflow.contrib.keras.python.keras.layers.pooling import GlobalAvgPool2D
from tensorflow.contrib.keras.python.keras.layers.pooling import GlobalAvgPool3D
from tensorflow.contrib.keras.python.keras.layers.pooling import GlobalMaxPool1D
from tensorflow.contrib.keras.python.keras.layers.pooling import GlobalMaxPool2D
from tensorflow.contrib.keras.python.keras.layers.pooling import GlobalMaxPool3D

# Recurrent layers.
from tensorflow.contrib.keras.python.keras.layers.recurrent import SimpleRNN
from tensorflow.contrib.keras.python.keras.layers.recurrent import GRU
from tensorflow.contrib.keras.python.keras.layers.recurrent import LSTM

# Wrapper layers.
from tensorflow.contrib.keras.python.keras.layers.wrappers import Wrapper
from tensorflow.contrib.keras.python.keras.layers.wrappers import Bidirectional
from tensorflow.contrib.keras.python.keras.layers.wrappers import TimeDistributed

del absolute_import
del division
del print_function

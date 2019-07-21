#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-6-5 下午12:03
# @Author  : hcq
# @File    : GruCapsule.py

import os
import re
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.callbacks import *
import warnings
warnings.filterwarnings('ignore')
from keras.engine.topology import Layer
from util import *


def Gru_Capsule_Model(sent_length, embeddings_weight,class_num):
    content = Input(shape=(sent_length,), dtype='int32')
    embedding = Embedding(
        name="word_embedding",
        input_dim=embeddings_weight.shape[0],
        weights=[embeddings_weight],
        output_dim=embeddings_weight.shape[1],
        trainable=False)
    embed = SpatialDropout1D(0.2)(embedding(content))
    x = Bidirectional(CuDNNGRU(200, return_sequences=True))(embed)
    capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings, share_weights=True)(x)
    capsule = Flatten()(capsule)
    x = Dense(1000)(capsule)
    x = BatchNormalization()(x)
    x = Activation(activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(500)(x)
    x = BatchNormalization()(x)
    x = Activation(activation="relu")(x)
    output = Dense(class_num, activation="softmax")(x)
    model = Model(inputs=content, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


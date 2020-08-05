
# coding: utf-8

# In[1]:


# coding:utf-8
import os
import re
import sys
import jieba
import pandas as pd
import numpy as np

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

import tensorflow as tf

import keras
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.callbacks import *
from keras.preprocessing import text, sequence
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split

import warnings
warnings.filterwarnings('ignore')

import os
import gc
import random
import feather




# In[3]:


path="../stacking/"
capsule_lstm_train_char=np.load(path+"capsule_lstm10_article.npz")['train']
capsule_lstm_test_char=np.load(path+"capsule_lstm10_article.npz")['test']
capsule_lstm_train_word=np.load(path+"capsule_lstm10_word_seg.npz")['train']
capsule_lstm_test_word=np.load(path+"capsule_lstm10_word_seg.npz")['test']
gru3_train_word=np.load(path+"get_text_gru310_word_seg.npz")['train']
gru3_test_word=np.load(path+"get_text_gru310_word_seg.npz")['test']
gru4_train_word=np.load(path+"get_text_gru410_word_seg.npz")['train']
gru4_test_word=np.load(path+"get_text_gru410_word_seg.npz")['test']
rcnn5_train_word=np.load(path+"get_text_rcnn510_word_seg.npz")['train']
rcnn5_test_word=np.load(path+"get_text_rcnn510_word_seg.npz")['test']
lstm_att_train_word=np.load(path+"get_text_lstm_attention10_word_seg.npz")['train']
lstm_att_test_word=np.load(path+"get_text_lstm_attention10_word_seg.npz")['test']
capsule_gru_train_word=np.load(path+"get_text_capsule10_word_seg.npz")['train']
capsule_gru_test_word=np.load(path+"get_text_capsule10_word_seg.npz")['test']
rcnn4_train_word=np.load(path+"get_text_rcnn410_word_seg.npz")['train']
rcnn4_test_word=np.load(path+"get_text_rcnn410_word_seg.npz")['test']
gru1_train_word=np.load(path+"get_text_gru110_word_seg.npz")['train']
gru1_test_word=np.load(path+"get_text_gru110_word_seg.npz")['test']
capsule_gru_train_char=np.load(path+"get_text_capsule10_article.npz")['train']
capsule_gru_test_char=np.load(path+"get_text_capsule10_article.npz")['test']
grulstm_att_train_char=np.load(path+"get_text_lstm_attention10_article.npz")['train']
grulstm_att_test_char=np.load(path+"get_text_lstm_attention10_article.npz")['test']


# In[4]:


x_train=np.concatenate([capsule_lstm_train_char,
                           capsule_lstm_train_word,
                           gru3_train_word,
                           gru4_train_word,
                           rcnn5_train_word,
                       lstm_att_train_word,
                       capsule_gru_train_word,
                       rcnn4_train_word,
                       capsule_gru_train_char,
                        grulstm_att_train_char
                       ],axis=1)

x_test=np.concatenate([capsule_lstm_test_char,
                           capsule_lstm_test_word,
                           gru3_test_word,
                           gru4_test_word,
                           rcnn5_test_word,
                       lstm_att_test_word,
                       capsule_gru_test_word,
                       rcnn4_test_word,
                       capsule_gru_test_char,
                       grulstm_att_test_char
                         ],axis=1)


import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adadelta
from itertools import product
import datetime


# In[8]:


train=feather.read_dataframe("../data/train_set.feather")
test=feather.read_dataframe("../data/test_set.feather")
result=test[['id']].copy()
train_label=train['class'].values
lb = LabelEncoder()
lb.fit(train['class'].values)


# In[9]:


nb_classes =19
dims = x_train.shape[1]
epochs = 15
# parameter grids
param_grid = [
     #(1, 6, 0.73, 0.756, 0.00001, 0.017, 2400),
     # (1, 8, 0.789, 0.97, 0, 0.018, 1100),
     #(1, 5, 0.7, 0.7, 0.001, 0.01, 1500),
     # (1, 6, 0.89, 0.994, 0.0001, 0.02421, 700),
     #(1, 10, 0.74, 0.908, 0.0005, 0.0141, 1750),
     #(1, 15, 0.7890, 0.890643, 0.231, 0.21, 900),
     #(1, 19, 0.78, 0.97453, 0.00009, 0.01, 3900),
     #(1, 6, 0.71, 0.71, 0, 0.01, 1250),
      #(1, 8, 0.77, 0.83, 0.001, 0.03, 900),
     (1, 3, 0.7, 0.7, 0.00008, 0.01, 300),
     #(1, 8, 0.824, 0.0241, 0.000177,0.02406 ,743)        

# kb8:  (1, 10, 0.87, 0.88, 0.000429, 0.029963, 652)        
# kb9:  (1, 8, 0.824, 0.0241, 0.000177,0.02406 ,743 )         

]



# In[10]:


from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=520).split(train['id'])
fold_index=train[['id']].copy()
for i, (train_fold, test_fold) in enumerate(kf):
    fold_index.loc[test_fold,'fold']=int(i)
#     print(len(test_fold))
#     break
fold_index['fold']=fold_index['fold'].astype(int)
fold_index.to_csv('../cache/fold_index.csv',index=False)





from sklearn.metrics import f1_score
import xgboost as xgb
xfolds = pd.read_csv('fold_index.csv')
# work with 5-fold split
fold_index = xfolds.fold
n_folds = len(np.unique(fold_index))
train_model_pred = np.zeros((x_train.shape[0], 19))
test_model_pred = np.zeros((x_test.shape[0], 19))
for i in range(len(param_grid)):
    print("processing parameter combo:", param_grid[i])
    # configure model with j-th combo of parameters
    x = param_grid[i]
    clf = xgb.XGBClassifier(objective='multi:softmax',
                            n_estimators=x[6],
                            max_depth=x[1],
                            min_child_weight=x[0],
                            learning_rate=x[5],
                            silent=True,
                            subsample=x[3],
                            colsample_bytree=x[2],
                            gamma=x[2],
                            seed=6666,
                            num_class=19,
                            n_jobs=10)
    for j in range(0,n_folds):
        idx0 = np.where(fold_index != j)
        idx1 = np.where(fold_index == j)  
        x0 = np.array(x_train)[idx0,:][0]
        x1 = np.array(x_train)[idx1,:][0]
        y0 = np.array(train_label)[idx0]
        y1 = np.array(train_label)[idx1]
        clf.fit(x0, y0, eval_metric="mlogloss", eval_set=[(x0, y0),(x1, y1)],verbose=100)

        train_model_pred[idx1, :] =  clf.predict_proba(x1)
        test_model_pred +=clf.predict_proba(x_test)
        print ("valid's accuracy: %s" % f1_score(y1.reshape(-1,1), 
                                                lb.inverse_transform(np.argmax( clf.predict_proba(x1), 1)).reshape(-1,1),
                                                              average='micro'))

        print("finished fold:", j)



# In[13]:


print ("offline test score: %s" % f1_score(train_label.reshape(-1,1), 
                                      lb.inverse_transform(np.argmax(train_model_pred, 1)).reshape(-1,1),
                                      average='micro'))

clf.fit(x_train, train_label, eval_metric="mlogloss",verbose=100)
test_model_pred =clf.predict_proba(x_test)


# In[14]:


np.savez('../satcking/stacking_offline8130664763338775.npz', train=train_model_pred, test=test_model_pred)


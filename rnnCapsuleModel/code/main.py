# coding: utf-8
import tensorflow as tf
import feather
import os
import re
import sys  
import gc
import random
import pandas as pd
import numpy as np
import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from scipy import stats
import keras
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.callbacks import *
from keras.preprocessing import text, sequence
from keras.utils import to_categorical
from keras.engine.topology import Layer
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.utils.training_utils import multi_gpu_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.metrics import  accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings('ignore')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--gpu",type=str)
parser.add_argument("--column_name",type=str)
parser.add_argument("--word_seq_len",type=int)
parser.add_argument("--embedding_vector",type=int)
parser.add_argument("--num_words",type=int)
parser.add_argument("--model_name",type=str)
parser.add_argument("--batch_size",type=int)
parser.add_argument("--KFold",type=int)
parser.add_argument("--classification",type=int)
args=parser.parse_args()



if not os.path.exists("../embedding"):
    os.mkdir("../embedding")

if not os.path.exists("../cache"):
    os.mkdir("../cache")

if not os.path.exists("../stacking"):
    os.mkdir("../stacking")


if not os.path.exists("../mid_result"):
    os.mkdir("../mid_result")


if not os.path.exists("../submission"):
    os.mkdir("../submission")



from GruCapsule import Gru_Capsule_Model
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


#导入数据
train=feather.read_dataframe("../data/train_set.feather")
test=feather.read_dataframe("../data/test_set.feather")



#词向量
def w2v_pad(df_train,df_test,col, maxlen_,victor_size):

    tokenizer = text.Tokenizer(num_words=args.num_words, lower=False,filters="")
    tokenizer.fit_on_texts(list(df_train[col].values)+list(df_test[col].values))

    train_ = sequence.pad_sequences(tokenizer.texts_to_sequences(df_train[col].values), maxlen=maxlen_)
    test_ = sequence.pad_sequences(tokenizer.texts_to_sequences(df_test[col].values), maxlen=maxlen_)
    
    word_index = tokenizer.word_index
    
    count = 0
    nb_words = len(word_index)
    print(nb_words)
    all_data=pd.concat([df_train[col],df_test[col]])
    file_name = '../embedding/' + 'Word2Vec_' + col  +"_"+ str(victor_size) + '.model'
    if not os.path.exists(file_name):
        model = Word2Vec([[word for word in document.split(' ')] for document in all_data.values],
                         size=victor_size, window=5, iter=10, workers=11, seed=2018, min_count=2)
        model.save(file_name)
    else:
        model = Word2Vec.load(file_name)
    print("add word2vec finished....")    


    glove_model = {}
    with open("../embedding/glove_vectors_word.txt",encoding='utf8') as f:
        for line in f:
            values = line.rstrip().rsplit(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            glove_model[word] = coefs
    print("add glove finished....")  
                 
    embedding_word2vec_matrix = np.zeros((nb_words + 1, victor_size))
    for word, i in word_index.items():
        embedding_vector = model[word] if word in model else None
        if embedding_vector is not None:
            count += 1
            embedding_word2vec_matrix[i] = embedding_vector
        else:
            unk_vec = np.random.random(victor_size) * 0.5
            unk_vec = unk_vec - unk_vec.mean()
            embedding_word2vec_matrix[i] = unk_vec


    glove_count=0
    embedding_glove_matrix = np.zeros((nb_words + 1, victor_size))
    for word, i in word_index.items():
        embedding_glove_vector=glove_model[word] if word in glove_model else None
        if embedding_glove_vector is not None:
            glove_count += 1
            embedding_glove_matrix[i] = embedding_glove_vector
        else:
            unk_vec = np.random.random(victor_size) * 0.5
            unk_vec = unk_vec - unk_vec.mean()
            embedding_glove_matrix[i] = unk_vec

    embedding_matrix=np.concatenate((embedding_word2vec_matrix,embedding_glove_matrix),axis=1)
    
    print (embedding_matrix.shape, train_.shape, test_.shape, count * 1.0 / embedding_matrix.shape[0],glove_count*1.0/embedding_matrix.shape[0])
    return train_, test_, word_index, embedding_matrix




word_seq_len=args.word_seq_len
victor_size=args.embedding_vector
column_name=args.column_name
train_, test_,word2idx, word_embedding = w2v_pad(train,test,column_name, word_seq_len,victor_size)



def word_model_cv(my_opt):
    #参数
    lb = LabelEncoder()
    train_label = lb.fit_transform(train['class'].values)
    train_label = to_categorical(train_label)

    if not os.path.exists("../cache/"+my_opt):
        os.mkdir("../cache/"+my_opt)

    #模型
    my_opt=eval(my_opt)
    name = str(my_opt.__name__)
    kf = KFold(n_splits=args.KFold, shuffle=True, random_state=520).split(train_)
    train_model_pred = np.zeros((train_.shape[0], args.classification))
    test_model_pred = np.zeros((test_.shape[0], args.classification))

    for i, (train_fold, test_fold) in enumerate(kf):
        X_train, X_valid, = train_[train_fold, :], train_[test_fold, :]
        y_train, y_valid = train_label[train_fold], train_label[test_fold]

        print(i, 'fold')

        the_path = '../cache/' + name +'/' +  name + "_" +args.column_name
        model = my_opt(word_seq_len, word_embedding,args.classification)
        early_stopping = EarlyStopping(monitor='val_acc', patience=6)
        plateau = ReduceLROnPlateau(monitor="val_acc", verbose=1, mode='max', factor=0.5, patience=3)
        checkpoint = ModelCheckpoint(the_path + str(i) + '.hdf5', monitor='val_acc', verbose=2, save_best_only=True, mode='max',save_weights_only=True)
        if not os.path.exists(the_path + str(i) + '.hdf5'):
            model.fit(X_train, y_train,
                      epochs=100,
                      batch_size=args.batch_size,
                      validation_data=(X_valid, y_valid),
                      callbacks=[early_stopping, plateau, checkpoint],
                      verbose=2)


        model.load_weights(the_path + str(i) + '.hdf5')

        print (name + ": valid's accuracy: %s" % f1_score(lb.inverse_transform(np.argmax(y_valid, 1)), 
                                                          lb.inverse_transform(np.argmax(model.predict(X_valid), 1)).reshape(-1,1),
                                                          average='micro'))
    
        train_model_pred[test_fold, :] =  model.predict(X_valid)
        test_model_pred += model.predict(test_)
        
        del model; gc.collect()
        K.clear_session()
    #线下测试
    print (name + ": offline test score: %s" % f1_score(lb.inverse_transform(np.argmax(train_label, 1)), 
                                                  lb.inverse_transform(np.argmax(train_model_pred, 1)).reshape(-1,1),
                                                  average='micro'))

    #中间结果记录
    mid_pred=test[['id']].copy()
    mid_pred=pd.concat([mid_pred,pd.DataFrame(test_model_pred)],axis=1)

    mid_pred.to_csv('../mid_result/{0}_KFold{1}_bs{2}_w2v{2}_len{3}_column_name{4}.csv'
                                                                        .format(name,
                                                                                args.KFold,
                                                                                args.batch_size,
                                                                                args.embedding_vector,
                                                                                args.word_seq_len,
                                                                                args.column_name
                                                                                ),index=False)
    
    last_pred=test[['id']].copy()
    last_pred['class']=lb.inverse_transform(np.argmax(test_model_pred, 1)).reshape(-1,1)
    last_pred[['id',"class"]].to_csv('../submission/{0}_KFold{1}_bs{2}_w2v{2}_len{3}_column_name{4}.csv'
                                                                                            .format(name,
                                                                                                    args.KFold,
                                                                                                    args.batch_size,
                                                                                                    args.embedding_vector,
                                                                                                    args.word_seq_len,
                                                                                                    args.column_name
                                                                                                    ),index=False)

    #
    test_model_pred /= args.KFold
    np.savez("../stacking/" + my_opt.__name__+ str(args.KFold) + '_' + args.column_name +'.npz', train=train_model_pred, test=test_model_pred)




word_model_cv(args.model_name)



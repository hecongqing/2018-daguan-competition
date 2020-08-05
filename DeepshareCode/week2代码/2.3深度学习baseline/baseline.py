import pandas as pd
import numpy as np
import tensorflow as tf
import os
from gensim.models import Word2Vec
from tensorflow.keras.layers import (Bidirectional,
                                     Embedding,
                                     GRU, 
                                     GlobalAveragePooling1D,
                                     GlobalMaxPooling1D,
                                     Concatenate,
                                     SpatialDropout1D,
                                     BatchNormalization,
                                     Dropout,
                                     Dense,
                                     Activation,
                                     concatenate,
                                     Input
                                    )
#读取数据集
train = pd.read_csv('../data/train_set.csv')
test = pd.read_csv('../data/test_set.csv')

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=50000, lower=False,filters="")
tokenizer.fit_on_texts(list(train['word_seg'].values)+list(test['word_seg'].values))

train_ = tokenizer.texts_to_sequences(train['word_seg'].values)
test_ = tokenizer.texts_to_sequences(test['word_seg'].values)

train_ = tf.keras.preprocessing.sequence.pad_sequences(train_, maxlen=1800)
test_ = tf.keras.preprocessing.sequence.pad_sequences(test_, maxlen=1800)

word_vocab = tokenizer.word_index


all_data=pd.concat([train['word_seg'],test['word_seg']])
file_name = '../embedding/Word2Vec_word_200_new.model'
if not os.path.exists(file_name):
    model = Word2Vec([[word for word in document.split(' ')] for document in all_data.values],
                     size=200, 
                     window=5,
                     iter=10, 
                     workers=11, 
                     seed=2018, 
                     min_count=2)

    model.save(file_name)
else:
    model = Word2Vec.load(file_name)
print("add word2vec finished....")    




count = 0

embedding_matrix = np.zeros((len(word_vocab) + 1, 200))
for word, i in word_vocab.items():
    embedding_vector = model.wv[word] if word in model else None
    if embedding_vector is not None:
        count += 1
        embedding_matrix[i] = embedding_vector
    else:
        unk_vec = np.random.random(200) * 0.5
        unk_vec = unk_vec - unk_vec.mean()
        embedding_matrix[i] = unk_vec
print("计算word embedding的覆盖词数：",count)


def build_model(sent_length, embeddings_weight,class_num):
    content = Input(shape=(sent_length,), dtype='int32')
    embedding = Embedding(
        name="word_embedding",
        input_dim=embeddings_weight.shape[0],
        weights=[embeddings_weight],
        output_dim=embeddings_weight.shape[1],
        trainable=False)

    x = SpatialDropout1D(0.2)(embedding(content))

    x = Bidirectional(GRU(200, return_sequences=True))(x)
    x = Bidirectional(GRU(200, return_sequences=True))(x)

    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)

    conc = concatenate([avg_pool, max_pool])

    x = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(1000)(conc))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))
    output = Dense(class_num, activation="softmax")(x)

    model = tf.keras.models.Model(inputs=content, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = build_model(1800, embedding_matrix,19)

model.summary()

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
lb = LabelEncoder()
train_label = lb.fit_transform(train['class'].values)
train_label = to_categorical(train_label)

from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val = train_test_split(train_,train_label,test_size=0.05,random_state=666)



X_train.shape,X_val.shape,y_train.shape,y_val.shape



train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000).batch(64)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(128)

from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint

# 检查点保存至的目录
checkpoint_dir = './training_checkpoints_bs64'

# 检查点的文件名
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

early_stopping = EarlyStopping(monitor='val_accuracy', patience=6)
plateau = ReduceLROnPlateau(monitor="val_accuracy", verbose=1, mode='max', factor=0.5, patience=3)
checkpoint = ModelCheckpoint(checkpoint_prefix, monitor='val_accuracy', verbose=2, save_best_only=True, mode='max',save_weights_only=True)

# model.fit(train_ds,
#           epochs=30,
#           validation_data=val_ds,
#           callbacks=[early_stopping, plateau, checkpoint],
#           verbose=2)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

valid_prob = model.predict(val_ds)
valid_pred = np.argmax(valid_prob,axis=1)
valid_pred = lb.inverse_transform(valid_pred)

val_true = np.argmax(y_val,axis=1)
val_true = lb.inverse_transform(val_true)

from sklearn.metrics import f1_score
print ("valid's macro_f1: %s" % f1_score(val_true,valid_pred,average='macro'))

test_ds = tf.data.Dataset.from_tensor_slices((test_,np.zeros((test_.shape[0],19)))).batch(128)
test_ds
test_prob = model.predict(test_ds)

test_pred = np.argmax(test_prob,axis=1)
test['class'] = lb.inverse_transform(test_pred)

test[["id","class"]].to_csv("submission_dnn_baseline.csv",index=False,header=True,encoding='utf-8')

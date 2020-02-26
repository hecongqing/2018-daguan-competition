import pandas as pd
import numpy as np
import tensorflow as tf
import os
import gc
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
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

#读取数据集
train = pd.read_csv('../data/train_set.csv')
test = pd.read_csv('../data/test_set.csv')

# import feather
# train=feather.read_dataframe("../data/train_set.feather")
# test=feather.read_dataframe("../data/test_set.feather")


tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=50000, lower=False,filters="")
tokenizer.fit_on_texts(list(train['word_seg'].values)+list(test['word_seg'].values))

train_ = tokenizer.texts_to_sequences(train['word_seg'].values)
test_ = tokenizer.texts_to_sequences(test['word_seg'].values)

train_ = tf.keras.preprocessing.sequence.pad_sequences(train_, maxlen=1800)
test_ = tf.keras.preprocessing.sequence.pad_sequences(test_, maxlen=1800)

word_vocab = tokenizer.word_index



from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
lb = LabelEncoder()
train_label = lb.fit_transform(train['class'].values)
train_label = to_categorical(train_label)


all_data=pd.concat([train['word_seg'],test['word_seg']])
file_name = '../embedding/Word2Vec_word_200.model'
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
        
def build_model(sent_length, embeddings_weight):
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

    x = Dense(1000)(conc)
    x = BatchNormalization()(x)
    x = Activation(activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(500)(x)
    x = BatchNormalization()(x)
    x = Activation(activation="relu")(x)
    output = Dense(19, activation="softmax")(x)

    model = tf.keras.models.Model(inputs=content, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model




def model_cv():
    kf = KFold(n_splits=10, shuffle=True, random_state=666)
    train_pre_matrix = np.zeros((train.shape[0],19)) #记录验证集的概率

    test_pre_matrix = np.zeros((10,test.shape[0],19)) #将10轮的测试概率分别保存起来
    cv_scores=[] #每一轮线下的验证成绩

    for i, (train_fold, test_fold) in enumerate(kf.split(train_)):
        print("第%s的结果"%i)
        X_train, X_valid = train_[train_fold, :], train_[test_fold, :]
        y_train, y_valid = train_label[train_fold], train_label[test_fold]
        
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(10000).batch(64)
        val_ds = tf.data.Dataset.from_tensor_slices((X_valid, y_valid)).batch(128)
        test_ds = tf.data.Dataset.from_tensor_slices((test_,np.zeros((test_.shape[0],19)))).batch(128)
        
        # 检查点保存至的目录
        checkpoint_dir = './cv_checkpoints1/cv_'+str(i)+'/'
        # 检查点的文件名
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
        model = build_model(1800, embedding_matrix)
        val_ds
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=6)
        plateau = ReduceLROnPlateau(monitor="val_accuracy", verbose=1, mode='max', factor=0.5, patience=3)
        checkpoint = ModelCheckpoint(checkpoint_prefix, monitor='val_accuracy', 
                                     verbose=2, save_best_only=True, mode='max',save_weights_only=True)

        model.fit(train_ds,
                  epochs=30,
                  validation_data=val_ds,
                  callbacks=[early_stopping, plateau, checkpoint],
                  verbose=2)


        model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
        
        valid_prob = model.predict(val_ds)
        valid_pred = np.argmax(valid_prob,axis=1)
        valid_pred = lb.inverse_transform(valid_pred)
        
        y_valid = np.argmax(y_valid, axis=1)
        y_valid = lb.inverse_transform(y_valid)
        
        f1_score_ = f1_score(y_valid,valid_pred,average='macro') 
        print ("valid's f1-score: %s" %f1_score_)
        
        
        train_pre_matrix[test_fold, :] =  valid_prob
        
        test_pre_matrix[i, :,:]= model.predict(test_ds)
        
        del model; gc.collect()
        tf.keras.backend.clear_session()    
    np.save("cv_test_result.npy",test_pre_matrix)

model_cv()

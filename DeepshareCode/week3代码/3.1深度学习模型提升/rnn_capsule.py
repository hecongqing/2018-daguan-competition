import pandas as pd
import numpy as np
import tensorflow as tf
import os
import gc
from gensim.models import Word2Vec
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Layer

from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

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



from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
lb = LabelEncoder()
train_label = lb.fit_transform(train['class'].values)
train_label = to_categorical(train_label)


all_data=pd.concat([train['word_seg'],test['word_seg']])
file_name = '../embedding/Word2Vec_word_200_new.model'
if not os.path.exists(file_name):
    model = Word2Vec([[word for word in document.split(' ')] for document in all_data.values],
                     size=vector_size, window=5, iter=10, workers=11, seed=2018, min_count=2)
    model.save(file_name)
else:
    model = Word2Vec.load(file_name)
print("add word2vec finished....")    


glove_model = {}
with open("../embedding/glove_vectors_vec.txt",encoding='utf-8') as f:
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        glove_model[word] = coefs
print("add glove finished....")  




vector_size =200

embedding_word2vec_matrix = np.zeros((len(word_vocab) + 1, vector_size))
for word, i in word_vocab.items():
    embedding_vector = model[word] if word in model else None
    if embedding_vector is not None:
        embedding_word2vec_matrix[i] = embedding_vector
    else:
        unk_vec = np.random.random(vector_size) * 0.5
        unk_vec = unk_vec - unk_vec.mean()
        embedding_word2vec_matrix[i] = unk_vec


glove_count=0
embedding_glove_matrix = np.zeros((len(word_vocab) + 1, vector_size))
for word, i in word_vocab.items():
    embedding_glove_vector=glove_model[word] if word in glove_model else None
    if embedding_glove_vector is not None:
        embedding_glove_matrix[i] = embedding_glove_vector
    else:
        unk_vec = np.random.random(vector_size) * 0.5
        unk_vec = unk_vec - unk_vec.mean()
        embedding_glove_matrix[i] = unk_vec

embedding_matrix=np.concatenate((embedding_word2vec_matrix,embedding_glove_matrix),axis=1)




Num_capsule=10
Dim_capsule=16
Routings=3


def squash(x, axis=-1):
    s_squared_norm = tf.keras.backend.sum(tf.keras.backend.square(x), axis, keepdims=True)
    scale = tf.keras.backend.sqrt(s_squared_norm + tf.keras.backend.epsilon())
    return x / scale


class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)
    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = tf.keras.backend.conv1d(u_vecs, kernel=self.W)
        else:
            u_hat_vecs = tf.keras.backend.local_conv1d(u_vecs, kernel=self.W, kernel_size=[1], strides=[1])

        batch_size = tf.shape(u_vecs)[0]
        input_num_capsule = tf.shape(u_vecs)[1]
        u_hat_vecs = tf.reshape(u_hat_vecs, [batch_size, input_num_capsule,self.num_capsule, self.dim_capsule])
        u_hat_vecs = tf.transpose(u_hat_vecs,perm=[0, 2, 1, 3])# final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]
        b = tf.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            b = tf.transpose(b, perm=[0, 2, 1])  # shape = [None, input_num_capsule, num_capsule] 
            c = tf.nn.softmax(b) # shape = [None, input_num_capsule, num_capsule] 
            c = tf.transpose(c, perm=[0, 2, 1])  # shape = [None, num_capsule, input_num_capsule] 
            s_j = tf.reduce_sum(tf.multiply(tf.expand_dims(c,axis=3) , u_hat_vecs) , axis=2)        
            outputs = self.activation(s_j) #[None,num_capsule,dim_capsule]
            if i < self.routings - 1:
                b = tf.reduce_sum(tf.multiply(tf.expand_dims(outputs,axis=2) , u_hat_vecs) , axis=3)
        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)

    
    
def Gru_Capsule_Model(sent_length, embeddings_weight,class_num):
    content = Input(shape=(sent_length,), dtype='int32')
    embedding = Embedding(
        name="word_embedding",
        input_dim=embeddings_weight.shape[0],
        weights=[embeddings_weight],
        output_dim=embeddings_weight.shape[1],
        trainable=False)
    embed = SpatialDropout1D(0.2)(embedding(content))
    x = Bidirectional(GRU(200, return_sequences=True))(embed)
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
        checkpoint_dir = './rnncapsule_cv_checkpoints/cv_'+str(i)+'/'
        # 检查点的文件名
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
        model = Gru_Capsule_Model(1800, embedding_matrix, 19)
        val_ds
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=6)
        plateau = ReduceLROnPlateau(monitor="val_accuracy", verbose=1, mode='max', factor=0.5, patience=3)
        checkpoint = ModelCheckpoint(checkpoint_prefix, monitor='val_accuracy', 
                                     verbose=2, save_best_only=True, mode='max',save_weights_only=True)

        if not os.path.exists(checkpoint_dir):
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

res = np.load("cv_test_result.npy")
res_mean = res.mean(axis=0)

test_pred = lb.inverse_transform(np.argmax(res_mean,axis=1))
test['class'] = test_pred
test[["id","class"]].to_csv("submission_baseline_capsule_cv.csv",index=False,header=True,encoding='utf-8')

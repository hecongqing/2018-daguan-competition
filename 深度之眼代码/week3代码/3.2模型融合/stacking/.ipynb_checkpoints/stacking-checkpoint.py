import lightgbm as lgb
import pandas as pd
import numpy as np

train=feather.read_dataframe("../data/train_set.feather")
test=feather.read_dataframe("../data/test_set.feather")




#####read model data################
capsule_train=np.load("../stacking/ xxxx")['train']
capsule_test=np.load("../stacking/ xxxx")['test']

lstm_attention_train=np.load("../stacking/ xxxx")['train']
lstm_attention_test=np.load("../stacking/ xxxx")['test']

train=np.concatenate([capsule_train,lstm_attention_train],axis=1)
test=np.concatenate([capsule_test,lstm_attention_test],axis=1)



params = {
        "objective": "binary",
        'metric': {'auc'},
        "boosting_type": "gbdt",
        "verbosity": -1,
        "num_threads": 4,
        "max_depth":3,
        "bagging_fraction": 0.8,
        "bagging_freq":5,
        #"colsample_bytree":0.45,
        "feature_fraction": 0.45,
        "learning_rate": 0.1,
        "num_leaves": 3,
        "verbose": -1,
        #"min_split_gain": .1,
        "reg_alpha": .3
    }


folds = KFold(n_splits=10, shuffle=True, random_state=233)

train_model_pred=np.zeros((train.shape[0],19))
test_model_pred=np.zeros((test.shape[0],19))

dtrain = lgb.Dataset(train, free_raw_data=False)
dtrain.set_label(train['class'].values)
lb = LabelEncoder()
train_label = lb.fit_transform(train['class'].values)
train_label = to_categorical(train_label)
for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train, train_label)):
    watchlist = [
        dtrain.subset(trn_idx),
        dtrain.subset(val_idx)
    ]
    model = lgb.train(
        params=params,
        train_set=watchlist[0],
        num_boost_round=1000,
        valid_sets=watchlist,
        early_stopping_rounds=50,
        verbose_eval=0
    )
    valid_pred=model.predict(dtrain.data[val_idx], num_iteration=model.best_iteration)
    train_model_pred[val_idx, :] =  valid_pred
    test_model_pred += model.predict(test)
    print(valid_pred.shape)
    print (name + ": valid's accuracy: %s" % f1_score(lb.inverse_transform(np.argmax(train_label[val_idx], 1)), 
                                                      lb.inverse_transform(np.argmax(valid_pred, 1)).reshape(-1,1),
                                                      average='micro'))

    print (name + ": offline test score: %s" % f1_score(lb.inverse_transform(np.argmax(train_label, 1)), 
                                                  lb.inverse_transform(np.argmax(train_model_pred, 1)).reshape(-1,1),
                                                  average='micro'))






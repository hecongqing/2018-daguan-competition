import feather
import pandas as pd


train=feather.read_dataframe("../data/train_set.feather")
test=feather.read_dataframe("../data/test_set.feather")

df=pd.concat([train['word_seg'],test['word_seg']])


with open("glove_word_vec.txt",'w') as f:
    for line in df.values:
        f.write(line+'\n')
f.close()
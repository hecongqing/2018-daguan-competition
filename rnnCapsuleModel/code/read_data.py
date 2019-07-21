#注意首先先安装feather包，pip install feather-format
import feather
import pandas as pd

def gen_csv_feather(path,path_new):
    f = open(path)
    reader = pd.read_csv(f, sep=',', iterator=True)
    loop = True
    chunkSize = 10000
    chunks = []
    while loop:
        try:
            chunk = reader.get_chunk(chunkSize)
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print("Iteration is stopped.")
    df = pd.concat(chunks, ignore_index=True)
    print(df.count())
    feather.write_dataframe(df,path_new)
    
gen_csv_feather("../data/train_set.csv","../data/train_set.feather")
gen_csv_feather("../data/test_set.csv","../data/test_set.feather")
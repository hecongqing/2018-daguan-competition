#! /bin/bash

CURDIR="`pwd`"/"`dirname $0`"

cd  $CURDIR/code

python read_data.py 


if [ ! -d "$CURDIR/embedding/" ]; then

  mkdir "$CURDIR/embedding/"

fi


if [ ! -f "$CURDIR/embedding/glove_vectors_char.txt" ]; then

  cd $CURDIR/glove

  python glove_char.py

  make

  sh $CURDIR/glove/glove_char.sh

  mv $CURDIR/glove/glove_vectors_char.txt  $CURDIR/embedding/

fi


if [ ! -f "$CURDIR/embedding/glove_vectors_word.txt" ]; then

  cd $CURDIR/glove

  python glove_word.py

  make

  sh $CURDIR/glove/glove_word.sh

  mv $CURDIR/glove/glove_vectors_word.txt   $CURDIR/embedding/

fi


cd $CURDIR/code/

CUDA_VISIBLE_DEVICES=0 python main_glove_word2vec.py  --gpu="0" --column_name="word_seg" --word_seq_len=1800 --embedding_vector=200 --num_words=500000 --model_name="bi_gru_model" --batch_size=128 --KFold=10 --classification=19



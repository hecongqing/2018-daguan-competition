CURDIR="`pwd`"/"`dirname $0`"
python $CURDIR/code/read_data.py

if [ ! -d "$CURDIR/embedding/" ]; then
  mkdir "$CURDIR/embedding/"
fi


if [ ! -f "$CURDIR/embedding/glove_vectors_char.txt" ]; then
  python $CURDIR/glove/glove_char.py
  sh $CURDIR/glove/glove_char.sh
  mv $CURDIR/glove/glove_vectors_char.txt  $CURDIR/embedding/
fi


if [ ! -f "$CURDIR/embedding/glove_vectors_word.txt" ]; then
  python $CURDIR/glove/glove_word.py
  sh $CURDIR/glove/glove_word.sh
  mv $CURDIR/glove/glove_vectors_word.txt   $CURDIR/embedding/
fi




CUDA_VISIBLE_DEVICES=0 python $CURDIR/code/main_glove_word2vec.py  --gpu="0" --column_name="word_seg" --word_seq_len=1800 --embedding_vector=200 --num_words=500000 --model_name="bi_gru_model" --batch_size=128 --KFold=10 --classification=19



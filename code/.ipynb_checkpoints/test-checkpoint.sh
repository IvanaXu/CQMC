
batch=32
python test/predict.py --device gpu --params_path ../user_data/model_data/model_state.pdparams --batch_size $batch

cd ../prediction_result
zip result6.zip bq_corpus.tsv lcqmc.tsv paws-x.tsv

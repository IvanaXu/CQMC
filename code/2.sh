
clear

dt=$(date "+%Y%m%d%H%M%S")

batch=16

cp ../user_data/model_data/model_state.pdparams ../user_data/model_data/model_state_$dt.pdparams
time python test/predict.py --device gpu --params_path ../user_data/model_data/model_state_$dt.pdparams --batch_size $batch

cd ../prediction_result
zip result6.zip bq_corpus.tsv lcqmc.tsv paws-x.tsv

mkdir $dt
zip download_$dt.zip predict.json result5.csv result6.zip
rm -rf $dt

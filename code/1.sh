
clear

rm -rf log/* ../user_data/model_data/* ../prediction_result/*
python train/cdata.py
md5sum ../user_data/cut_data/*.csv > ../user_data/cut_data/md5

batch=200
time python -m paddle.distributed.launch --device gpu --gpus "0,1,2,3" train/train.py --save_dir ../user_data --epochs 24 --batch_size $batch

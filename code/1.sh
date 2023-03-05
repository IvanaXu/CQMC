
clear

rm -rf log/* ../user_data/model_data/* ../prediction_result/*
python train/cdata.py
md5sum ../user_data/cut_data/*.csv > ../user_data/cut_data/md5

batch=256
time python -m paddle.distributed.launch --device gpu --gpus "0" train/train.py --save_dir ../user_data --epochs 100 --batch_size $batch

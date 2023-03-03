
clear

rm -rf log/* ../user_data/model_data/* ../prediction_result/*
python train/cdata.py

batch=180
time python -m paddle.distributed.launch --gpus "0" train/train.py --device gpu --save_dir ../user_data --epochs 24 --batch_size $batch

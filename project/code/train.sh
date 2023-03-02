
python train/cdata.py

batch=100
python -m paddle.distributed.launch --gpus "0" train/train.py --device gpu --save_dir ../user_data --epochs 24 --batch_size $batch

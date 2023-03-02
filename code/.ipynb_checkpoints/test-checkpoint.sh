batch=100
python test/predict.py --device gpu --params_path ../user_data/model_data/model_state.pdparams --batch_size $batch

#!/bin/sh
python3 vpg.py --env HalfCheetah-v2 --seed 1 --hidden-layers 64 32 --pi-lr 3e-4 --v-lr 1e-3 \
	--epochs 100 --steps-per-epoch 5000 --train-v-iters 80 --max-ep-len 1000 --gamma 0.99 --lamb 0.97
# python3 vpg.py --env HalfCheetah-v2 --seed 2 --cpu 4 --hidden-layers 64 64 --pi-lr 3e-4 --v-lr 1e-3 \
# 	--epochs 100 --steps-per-epoch 5000 --train-v-iters 80 --max-ep-len 1000 --gamma 0.99 --lamb 0.97
# python3 vpg.py --env HalfCheetah-v2 --seed 3 --cpu 4 --hidden-layers 64 64 --pi-lr 3e-4 --v-lr 1e-3 \
# 	--epochs 100 --steps-per-epoch 5000 --train-v-iters 80 --max-ep-len 1000 --gamma 0.99 --lamb 0.97
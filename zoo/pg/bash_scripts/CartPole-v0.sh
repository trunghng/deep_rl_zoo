#!/bin/sh
cd ..
python3 main.py --env CartPole-v0 --seed 1 --pi-lr 7e-4 --v-lr 1e-3 \
	--epochs 500 --steps-per-epoch 2500 --train-v-iters 20 \
	--max-ep-len 300 --gamma 0.99 --lamb 0.96 --goal 195 --save --render
python3 main.py --eval --model-path ./output/models/CartPole-v0 --env CartPole-v0
#!/bin/sh
python3 main.py --env CartPole-v0 --seed 1 --pi-lr 8e-3 --v-lr 3e-3 \
	--epochs 50 --steps-per-epoch 5000 --train-v-iters 20 \
	--max-ep-len 500 --gamma 0.98 --lamb 0.96 --goal 200 --save --render
python3 main.py --eval --model-path ./output/models/VPG-CartPole-v0.pth --env CartPole-v0
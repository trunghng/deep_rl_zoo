#!/bin/sh
python3 trpo.py --env CartPole-v0 --cpu 4 --hidden-layers 64 --v-lr 1e-4 --epochs 100 --steps-per-epoch 5000\
	--train-v-iters 20 --max-ep-len 500 --gamma 1 --lamb 1 --delta 0.01 --damping-coeff 0.1 --cg-iters 10\
	--linesearch --backtrack-coeff 0.8 --backtrack-iters 10 --seed 1
# python3 trpo.py --eval --model-path ./output/models/TRPO-CartPole-v0.pth --env CartPole-v0

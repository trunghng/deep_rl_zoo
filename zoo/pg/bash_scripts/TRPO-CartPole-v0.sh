#!/bin/sh
python3 trpo.py --env CartPole-v0 --v-lr 1e-3 --epochs 50 --steps-per-epoch 5000\
	--train-v-iters 80 --max-ep-len 1000 --gamma 0.99 --lamb 0.97 --goal 200\
	--delta 0.01 --damping-coeff 0.1 --cg-iters 10 --linesearch --backtrack-coeff 0.8\
	--backtrack-iters 10 --seed 1 --save --render

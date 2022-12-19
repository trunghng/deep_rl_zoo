#!/bin/sh
python3 ppo.py --env CartPole-v0 --hidden-layers 32 32 --pi-lr 1e-2 --v-lr 1e-2 --epochs 500\
	--steps-per-epoch 2500 --train-pi-iters 40 --train-v-iters 40 --max-ep-len 500\
	--gamma 1 --lamb 1 --clip --clip-ratio 0.2 --kl-target 0.01 --seed 2
# python3 ppo.py --eval --model-path ./output/models/PPO-CartPole-v0.pth --env CartPole-v0
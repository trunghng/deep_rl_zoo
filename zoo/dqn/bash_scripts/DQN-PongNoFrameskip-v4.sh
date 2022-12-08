#!/bin/sh
python3 dqn.py --atari --env PongNoFrameskip-v4  --seed 1 --epsilon-init 1\
	--epsilon-final 0.01 --epsilon-decay 100 --gamma 0.99 --lr 1e-4\
	--buffer-size 100000 --batch-size 64 --train-freq 4 --update-target 1000\
	--n-eps 600 --log-int 10 --goal 20 --print-freq 10 --save --render --plot
python3 dqn.py --atari --env PongNoFrameskip-v4 --eval\
	--model-path ./output/models/Conv-DeepQ-ExpRep-PongNoFrameskip-v4.pth --seed 1
#!/bin/sh
python3 dqn.py --env LunarLander-v2 --epsilon-init 1 --epsilon-final 0.01\
	--epsilon-decay 200 --gamma 0.99 --lr 5e-4 --buffer-size 100000 --batch-size 64\
	--train-freq 4 --update-target 4 --tau 1e-3 --n-eps 1000 --log-int 100\
	--goal 200 --print-freq 100 --save --render --plot --seed 1
python3 dqn.py --env LunarLander-v2 --seed 1 --eval\
	--model-path ./output/models/MLP-DeepQ-ExpRep-LunarLander-v2.pth
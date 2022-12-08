#!/bin/sh
python3 dqn.py --env LunarLander-v2 --double --epsilon-init 1 --epsilon-final 0.01\
	--epsilon-decay 200 --gamma 0.99 --lr 5e-4 --buffer-size 100000 --batch-size 64\
	--train-freq 4 --update-target 4 --tau 1e-3 --n-eps 1000 --log-int 100\
	--goal 200 --print-freq 100 --save --render --plot --seed 2
python3 dqn.py --env LunarLander-v2 --double --seed 2 --eval\
	--model-path ./output/models/MLP-DoubleDeepQ-ExpRep-LunarLander-v2.pth
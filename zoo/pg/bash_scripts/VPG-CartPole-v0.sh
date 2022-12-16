#!/bin/sh
python3 vpg.py --env CartPole-v0 --seed 21 --cpu 4 --hidden-layers 64 --pi-lr 1e-2 --v-lr 1e-2 \
	--epochs 100 --steps-per-epoch 5000 --train-v-iters 1 --max-ep-len 500 --gamma 1 \
	--lamb 1
python3 vpg.py --env CartPole-v0 --seed 11 --cpu 4 --hidden-layers 64 --pi-lr 1e-2 --v-lr 1e-2 \
	--epochs 100 --steps-per-epoch 5000 --train-v-iters 1 --max-ep-len 500 --gamma 1 \
	--lamb 1
python3 vpg.py --env CartPole-v0 --seed 12 --cpu 4 --hidden-layers 64 --pi-lr 1e-2 --v-lr 1e-2 \
	--epochs 100 --steps-per-epoch 5000 --train-v-iters 1 --max-ep-len 500 --gamma 1 \
	--lamb 1
python3 vpg.py --env CartPole-v0 --seed 13 --cpu 4 --hidden-layers 64 --pi-lr 1e-2 --v-lr 1e-2 \
	--epochs 100 --steps-per-epoch 5000 --train-v-iters 1 --max-ep-len 500 --gamma 1 \
	--lamb 1
python3 vpg.py --env CartPole-v0 --seed 14 --cpu 4 --hidden-layers 64 --pi-lr 1e-2 --v-lr 1e-2 \
	--epochs 100 --steps-per-epoch 5000 --train-v-iters 1 --max-ep-len 500 --gamma 1 \
	--lamb 1
python3 vpg.py --env CartPole-v0 --seed 15 --cpu 4 --hidden-layers 64 --pi-lr 1e-2 --v-lr 1e-2 \
	--epochs 100 --steps-per-epoch 5000 --train-v-iters 1 --max-ep-len 500 --gamma 1 \
	--lamb 1
python3 vpg.py --env CartPole-v0 --seed 16 --cpu 4 --hidden-layers 64 --pi-lr 1e-2 --v-lr 1e-2 \
	--epochs 100 --steps-per-epoch 5000 --train-v-iters 1 --max-ep-len 500 --gamma 1 \
	--lamb 1
python3 vpg.py --env CartPole-v0 --seed 17 --cpu 4 --hidden-layers 64 --pi-lr 1e-2 --v-lr 1e-2 \
	--epochs 100 --steps-per-epoch 5000 --train-v-iters 1 --max-ep-len 500 --gamma 1 \
	--lamb 1
python3 vpg.py --env CartPole-v0 --seed 18 --cpu 4 --hidden-layers 64 --pi-lr 1e-2 --v-lr 1e-2 \
	--epochs 100 --steps-per-epoch 5000 --train-v-iters 1 --max-ep-len 500 --gamma 1 \
	--lamb 1
python3 vpg.py --env CartPole-v0 --seed 19 --cpu 4 --hidden-layers 64 --pi-lr 1e-2 --v-lr 1e-2 \
	--epochs 100 --steps-per-epoch 5000 --train-v-iters 1 --max-ep-len 500 --gamma 1 \
	--lamb 1
python3 vpg.py --env CartPole-v0 --seed 20 --cpu 4 --hidden-layers 64 --pi-lr 1e-2 --v-lr 1e-2 \
	--epochs 100 --steps-per-epoch 5000 --train-v-iters 1 --max-ep-len 500 --gamma 1 \
	--lamb 1
# python3 vpg.py --eval --model-path ./output/models/VPG-CartPole-v0.pth --env CartPole-v0
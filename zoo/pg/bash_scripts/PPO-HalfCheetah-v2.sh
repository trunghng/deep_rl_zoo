#!/bin/sh
# python3 ppo.py --env HalfCheetah-v2 --hidden-layers 8 8 --pi-lr 3e-4 --v-lr 1e-3 --epochs 100\
# 	--steps-per-epoch 4000 --train-pi-iters 80 --train-v-iters 80 --max-ep-len 1000\
# 	--gamma 0.99 --lamb 0.97 --clip --clip-ratio 0.2 --kl-target 0.01 --seed 1

# python3 ppo.py --env HalfCheetah-v2 --hidden-layers 64 32 --pi-lr 3e-4 --v-lr 1e-3 --epochs 100\
# 	--steps-per-epoch 4000 --train-pi-iters 80 --train-v-iters 80 --max-ep-len 1000\
# 	--gamma 0.99 --lamb 0.97 --clip --clip-ratio 0.2 --kl-target 0.01 --seed 1

# final
python3 ppo.py --env HalfCheetah-v2 --hidden-layers 64 32 --epochs 750 --save --render
# python3 ppo.py --eval --model-path ./output/models/PPO-HalfCheetah-v2.pth --env HalfCheetah-v2 --hidden-layers 64 32
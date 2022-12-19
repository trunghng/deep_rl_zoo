#!/bin/sh
python3 trpo.py --env HalfCheetah-v2 --hidden-layers 64 32 --epochs 200 --save --render
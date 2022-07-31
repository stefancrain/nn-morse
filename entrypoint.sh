#!/bin/bash
main.py --save 10 --total 1000 &
tensorboard --bind_all --logdir='/project/runs/' --port=5000

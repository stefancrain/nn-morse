#!/bin/bash
main.py --max 5000 &
tensorboard --bind_all --logdir='/project/runs/' --port=5000

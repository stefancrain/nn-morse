#!/bin/bash
jupyter notebook &
tensorboard --bind_all --logdir='/project/runs/' --port=5000

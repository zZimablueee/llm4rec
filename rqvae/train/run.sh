#!/bin/bash

# show ${RUNTIME_SCRIPT_DIR}
echo ${RUNTIME_SCRIPT_DIR}
# enter train workspace
cd ${RUNTIME_SCRIPT_DIR}

# write your code below
# python -u rqvae_train.py


python -u rqvae_train.py --dataset electronic --device cuda:0 --lr 0.001 --epochs 500

# cd /root/autodl-tmp/qwenrec/rqvae/train
# python -u rqvae_train.py


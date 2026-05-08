#!/bin/bash

NUM_GPUS=3

torchrun \
  --nproc_per_node=$NUM_GPUS \
  scripts/train.py --phase 1
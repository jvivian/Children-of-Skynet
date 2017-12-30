#!/usr/bin/env bash

# Define vars
FILE="data/sanitized-tweets.txt"
BASE_CMD="train-rnn --gpu-growth"
CMD="$BASE_CMD $FILE"

# Default parameters
${CMD}

# Individual parameter adjustments
${CMD} --units=512
${CMD} --num-layers=2
${CMD} --dropout=0.5
${CMD} --batch-size=512
${CMD} --maxlen=100
${CMD} --epochs=30
${CMD} --gru

# Parameter combinations
${CMD} --units=512 \
        --num-layers=2 \
        --maxlen=100 \
        --dropout=0.5

# Simple Model
${CMD} --units=128 \
        --num-layers=1 \
        --maxlen=10 \
        --dropout=0.2 \
        --epochs=20 \
        --gru \
        --batch-size=100

# Karpathy blog build
# http://karpathy.github.io/2015/05/21/rnn-effectiveness/
${CMD} --units=512 \
        --num-layers 3 \
        --maxlen=100 \
        --dropout=0.5

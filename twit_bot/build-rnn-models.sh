#!/usr/bin/env bash
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

# Parameter combinations
${CMD} --units=512 \
        --num-layers=2 \
        --maxlen=100 \
        --dropout=0.5

#!/bin/bash
export HF_HOME=../../.cache
LD_LIBRARY_PATH=""
python pretrain-gpt-unigram.py \
--output-dir ../../babylm-models/baseline-nocl-gpt-small \
--max-positions 512 \
--embed-dimension 768 \
--layers 12 \
--attention-heads 12 \
--lr 3e-5 \
--bsz 64 \
--save-steps 10000 \
--eval-steps 10000 \
--warmup-steps 5000 \
--max-steps 50000

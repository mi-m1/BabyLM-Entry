#!/bin/bash
export HF_HOME=../../.cache

LD_LIBRARY_PATH=""
python pretrain-baseline-cl-random.py \
--tc-scheme cl-baseline-random \
--output-dir ../../babylm-models/cl-baseline-random-gpt \
--max-positions 512 \
--embed-dimension 768 \
--layers 12 \
--attention-heads 12 \
--lr 3e-5 \
--bsz 64 \
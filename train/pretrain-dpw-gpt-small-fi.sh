#!/bin/bash

export HF_HOME=../../.cache
LD_LIBRARY_PATH=""
python pretrain-dpw-gpt-small-fi.py \
--tc-scheme dpw \
--output-dir ../../babylm-models/cl-dpw-gpt-small-fi \
--max-positions 512 \
--embed-dimension 768 \
--layers 12 \
--attention-heads 12 \
--lr 3e-5 \
--bsz 64 \
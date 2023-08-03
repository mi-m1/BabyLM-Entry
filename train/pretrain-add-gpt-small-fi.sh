#!/bin/bash
export HF_HOME=/mnt/parscratch/users/acq22zm/.cache

LD_LIBRARY_PATH=""
python pretrain-tc-gpt-small-fi.py \
--tc-scheme add \
--output-dir ../../babylm-models/cl-add-gpt-small-fi \
--max-positions 512 \
--embed-dimension 768 \
--layers 12 \
--attention-heads 12 \
--lr 3e-5 \
--bsz 64 \
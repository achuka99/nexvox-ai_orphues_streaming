#!/bin/bash
set -e

export HF_HOME="/workspace/hf"
export HF_TOKEN=""
export TRANSFORMERS_OFFLINE=0 


CUDA_VISIBLE_DEVICES=0 vllm serve crestai/english_uganda-orpheus-3b-finetuned \
--max-model-len 2048 \
--gpu-memory-utilization 0.85 \
--host 0.0.0.0 \
--port 9191 \
--max-num-batched-tokens 8192 \
--max-num-seqs 24 \
--enable-chunked-prefill \
--disable-log-requests \
--block-size 16 \
--enable-prefix-caching \
--quantization fp8 &


CUDA_VISIBLE_DEVICES=1 uvicorn main:app \
--host 0.0.0.0 \
--port 9090 \
--workers 24 \
--loop uvloop \
--http httptools \
--log-level warning \
--access-log
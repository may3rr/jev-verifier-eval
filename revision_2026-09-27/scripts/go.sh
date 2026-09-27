#!/bin/bash
# On the rented A100: wait for the model download, serve Qwen3.5-9B, run the revision jobs.
cd /root
until grep -q DLDONE dl.log; do sleep 20; done
grep -q MISMATCH dl.log && { echo "SHA MISMATCH, abort"; exit 1; }
pkill -f "[v]llm serve"; sleep 3
nohup vllm serve /root/models/Qwen3.5-9B --served-model-name Qwen/Qwen3.5-9B --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.92 \
  --max-num-batched-tokens 8192 --max-num-seqs 256 --max-logprobs 20 > vllm.log 2>&1 &
until curl -sf localhost:8000/v1/models | grep -q "Qwen/Qwen3.5-9B"; do sleep 10; done
echo "SERVER_UP $(date)"
cd /root/rev && SILICONFLOW_API_KEY=x SILICONFLOW_BASE_URL=http://127.0.0.1:8000/v1/chat/completions \
  python3 run_revision.py jobs.json out

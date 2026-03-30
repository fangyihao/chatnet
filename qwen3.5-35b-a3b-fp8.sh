CUDA_VISIBLE_DEVICES=1 vllm serve --enforce-eager Qwen/Qwen3.5-35B-A3B-FP8 --port 8000 --tensor-parallel-size 1 --max-model-len 8192 --reasoning-parser qwen3 --language-model-only

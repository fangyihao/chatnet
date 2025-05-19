CUDA_VISIBLE_DEVICES=0,2,3 vllm serve models/qwq:32b --served-model-name qwq:32b --max-model-len 4096 --max-num-seqs 20 --pipeline-parallel-size 3

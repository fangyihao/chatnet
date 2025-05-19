'''
Created on May 19, 2025

@author: Yihao Fang
'''
import subprocess
import os

def serve_vllm(model, env, **kwargs):
    """
    Serves a vLLM model using Python's subprocess module.

    Args:
        **kwargs: Additional arguments to pass to the vllm serve command.
    """
    # command = [sys.executable, "-m", "vllm.entrypoints.api_server"]
    command = ["vllm", "serve", model]

    for key, value in kwargs.items():
        command.extend([f"--{key.replace('_', '-')}", str(value)])

    process = subprocess.Popen(command, env=env)
    return process

def serve_node(model, ports, devices):
    for device, port in zip(devices, ports):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = device
        process = serve_vllm(model=f"models/{model}", env=env, served_model_name=model, host="0.0.0.0", port=port, dtype="half", max_model_len=4096, max_num_seqs=64, pipeline_parallel_size=1, gpu_memory_utilization=0.95)
        print(f"vLLM server started for model '{model}' with PID: {process.pid}")
    
if __name__ == "__main__":
    serve_node(model= "qwen2.5:14b", ports=[8001,8002,8003], devices=["0","2","3"])
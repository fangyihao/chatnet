'''
Created on May 17, 2025

@author: Yihao Fang
'''
import subprocess
import sys
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

# Example usage:
if __name__ == "__main__":
    model = "qwen2.5:14b"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "3"
    process = serve_vllm(model=f"models/{model}", env=env, served_model_name=model, host="0.0.0.0", port=8000, dtype="half", max_model_len=4096, max_num_seqs=64, pipeline_parallel_size=1, gpu_memory_utilization=0.95)
    print(f"vLLM server started for model '{model}' with PID: {process.pid}")
    '''
    try:
        process.wait()
    except KeyboardInterrupt:
        print("Stopping vLLM server...")
        process.terminate()
        process.wait()
        print("vLLM server stopped.")
    '''
    
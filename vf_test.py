import subprocess
import time

import modal

app = modal.App(name="verifiers")

cuda_version = "12.6.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

base_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .apt_install("git", "libibverbs-dev", "libibverbs1")
    .workdir("/root")
    .run_commands(
        "git clone https://github.com/willccbb/verifiers.git",
    )
    .workdir("verifiers")
    .run_commands(
        "uv sync",
        "uv pip install -e .",
        "uv pip install flash-attn --no-build-isolation",
    )
    .pip_install("requests")
)

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


@app.function(
    image=base_image,
    gpu="H100:8",
    secrets=[modal.Secret.from_name("huggingface-secret"), modal.Secret.from_name("wandb-secret")],
    timeout=60 * 60 * 24,
)
def run_grpo():
    import requests
    from requests.exceptions import ConnectionError

    cmd = [
        "source .venv/bin/activate",
        "&&",
        "CUDA_VISIBLE_DEVICES=0,1,2,3",
        "python",
        "verifiers/inference/vllm_serve.py",
        "--model",
        MODEL_ID,
        "--tensor_parallel_size",
        "4",
        "--max_model_len",
        "8192",
        "--gpu_memory_utilization",
        "0.9",
        "--enable_prefix_caching",
        "True",
    ]

    generate_process = subprocess.Popen(
        ["/bin/bash", "-c", " ".join(cmd)],
        stderr=subprocess.STDOUT,
    )

    print("Waiting for vLLM server to start...")
    while True:
        try:
            response = requests.get("http://localhost:8000/")
            if response.status_code:
                print("vLLM server is ready!")
                break
        except ConnectionError:
            pass
        time.sleep(1)

    cmd = [
        "source .venv/bin/activate",
        "&&",
        "CUDA_VISIBLE_DEVICES=4,5,6,7",
        "accelerate",
        "launch",
        "--num-processes",
        "4",
        "--config-file",
        "configs/zero3.yaml",
        "verifiers/examples/math_train.py",
    ]

    train_process = subprocess.Popen(
        ["/bin/bash", "-c", " ".join(cmd)],
        stderr=subprocess.STDOUT,
    )

    train_process.wait()
    print("\033[91mTrain process exited\033[0m")
    generate_process.wait()

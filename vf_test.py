import modal
import modal.experimental
import subprocess
import time

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
        "git clone https://github.com/pawalt/verifiers.git",
    )
    .workdir("verifiers")
    .run_commands(
        "uv sync",
        "uv pip install -e .",
        "uv pip install flash-attn --no-build-isolation",
    )
    .run_commands(
        "sed -i 's/AF_INET/AF_INET6/g' /root/verifiers/.venv/lib/python3.12/site-packages/vllm/distributed/utils.py"
    )
    .pip_install("requests")
    .add_local_file("math_train.py", "/root/verifiers/math_train.py")
)

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


def run_vllm_server(cluster_info):
    """Runs the vLLM server on rank 1."""

    rank = cluster_info.rank
    container_ips = cluster_info.container_ips
    vllm_server_ip = container_ips[1]  # Should be its own IP
    print(
        f"Container rank {rank} (vLLM Server) starting up. Will listen on {vllm_server_ip}:8000"
    )
    cmd = [
        "source .venv/bin/activate",
        "&&",
        "python",
        "verifiers/inference/vllm_serve.py",
        "--model",
        MODEL_ID,
        "--tensor_parallel_size",
        "4",  # Matches number of GPUs
        "--max_model_len",
        "8192",
        "--gpu_memory_utilization",
        "0.9",
        "--enable_prefix_caching",
        "True",
        "--host",  # Make server accessible from other containers
        "::",  # Use IPv6 address
    ]
    vllm_process = subprocess.Popen(
        ["/bin/bash", "-c", " ".join(cmd)],
        stderr=subprocess.STDOUT,
    )
    print(
        f"Container rank {rank} (vLLM Server) process started. Waiting for it to exit."
    )
    vllm_process.wait()
    print(f"\033[91mContainer rank {rank} (vLLM Server) process exited.[0m")


def run_trainer(cluster_info):
    """Runs the training process on rank 0, waiting for the vLLM server."""
    import requests
    from requests.exceptions import ConnectionError

    rank = cluster_info.rank
    container_ips = cluster_info.container_ips
    vllm_server_host_ip = container_ips[1]  # IP of the rank 1 container
    vllm_server_url = f"http://[{vllm_server_host_ip}]:8000/"
    print(
        f"Container rank {rank} (Trainer) waiting for vLLM server at {vllm_server_url}..."
    )

    while True:
        try:
            response = requests.get(vllm_server_url)  # Use the dynamic IP
            if response.status_code:  # Any status code means it's up
                print(f"Container rank {rank} (Trainer): vLLM server is ready!")
                break
        except ConnectionError:
            pass
        time.sleep(5)

    train_cmd = [
        "source .venv/bin/activate",
        "&&",
        "accelerate",
        "launch",
        "--num-processes",
        "4",
        "--config-file",
        "configs/zero3.yaml",
        "math_train.py",
        "--vllm_server_host",
        f"{vllm_server_host_ip}",
    ]

    train_process = subprocess.Popen(
        ["/bin/bash", "-c", " ".join(train_cmd)],
        stderr=subprocess.STDOUT,
    )

    train_process.wait()
    print(f"\033[91mContainer rank {rank} (Trainer) process exited.[0m")


@app.function(
    image=base_image,
    gpu="H100:8",
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
    timeout=60 * 60 * 24,
    cloud="oci",
)
@modal.experimental.clustered(2, rdma=True)
def run_grpo():
    cluster_info = modal.experimental.get_cluster_info()
    rank = cluster_info.rank

    if rank == 1:
        run_vllm_server(cluster_info)
    elif rank == 0:
        run_trainer(cluster_info)
    else:
        print(f"Container rank {rank} has no specific role in this setup.")

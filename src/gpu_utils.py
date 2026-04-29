# GPU selection utilities.
#
# Author: Grégory Sainton
# Institution: Observatoire de Paris - PSL University

import sys
import subprocess

import torch


def get_gpu_usage():
    """
    Return the raw output of nvidia-smi as a string.

    Returns:
        str: nvidia-smi stdout.
    """
    try:
        result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        return result.stdout
    except Exception as e:
        sys.exit(f"Failed to execute nvidia-smi: {e}")


def is_gpu_free(gpu_id):
    """
    Check whether a GPU has no running processes according to nvidia-smi.

    Args:
        gpu_id (int): Index of the GPU to check.

    Returns:
        bool: True if no processes are found on that GPU.
    """
    gpu_usage = get_gpu_usage()
    return "No running processes found" in gpu_usage.split("GPU")[gpu_id + 1]


def setup_device(device_str):
    """
    Validate and return a torch.device for the requested GPU or CPU.

    Exits with an error if the requested GPU index is out of range.
    Warns (but still uses) a GPU that appears busy.

    Args:
        device_str (str): e.g. 'cuda:0' or 'cpu'.

    Returns:
        torch.device: Selected device.
    """
    if not torch.cuda.is_available():
        print("No GPU available — using CPU.")
        return torch.device("cpu")

    n_gpus = torch.cuda.device_count()
    valid = [f"cuda:{i}" for i in range(n_gpus)]
    for i in range(n_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    if device_str not in valid:
        print(f"Invalid device '{device_str}'. Valid options: {valid}")
        sys.exit(1)

    gpu_id = int(device_str.split(":")[1])
    if not is_gpu_free(gpu_id):
        print(f"Warning: GPU {device_str} may already be in use.")

    print(f"Using device: {device_str}")
    return torch.device(device_str)

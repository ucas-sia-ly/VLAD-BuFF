"""Optional per-process PyTorch allocation ceiling; FAISS is monitored separately."""
import os
import torch


def configure_cuda_budget(device):
    limit = os.environ.get('VLADBUFF_GPU_ALLOC_GIB')
    if limit and device == 'cuda':
        total = torch.cuda.get_device_properties(0).total_memory
        budget = float(limit) * 1024 ** 3
        if budget > total:
            raise RuntimeError('Configured GPU budget exceeds physical VRAM')
        torch.cuda.set_per_process_memory_fraction(budget / total, device=0)

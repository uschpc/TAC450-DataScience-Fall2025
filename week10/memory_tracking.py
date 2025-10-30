import subprocess
import torch
import matplotlib.pyplot as plt
import numpy as np

def get_nvidia_smi_gpu_memory(device_id=0):
    """Get GPU memory usage from nvidia-smi for a specific GPU"""
    result = subprocess.check_output(
        [
            'nvidia-smi',
            '--query-gpu=memory.used',
            '--format=csv,nounits,noheader',
            f'--id={device_id}'
        ], encoding='utf-8'
    )
    return float(result.strip())

def get_memory_breakdown(model, optimizer, device_id=0):
    """Returns memory usage in MB for different categories"""
    memory = {}
    
    # Basic memory stats
    memory['allocated'] = torch.cuda.memory_allocated(device_id) / 1024 / 1024
    memory['cached'] = torch.cuda.memory_reserved(device_id) / 1024 / 1024
    memory['nvidia_smi'] = get_nvidia_smi_gpu_memory(device_id)
    
    # Track parameters for this device only
    params_on_device = [p for p in model.parameters() if p.device.index == device_id]
    
    memory['weights'] = sum(p.numel() * p.element_size() for p in params_on_device) / 1024 / 1024
    memory['gradients'] = sum(p.grad.numel() * p.grad.element_size() 
                            for p in params_on_device 
                            if p.grad is not None) / 1024 / 1024
    
    # Optimizer state memory
    memory['optimizer'] = 0
    for p in params_on_device:
        state = optimizer.state.get(p, {})
        if state:
            for value in state.values():
                if torch.is_tensor(value):
                    memory['optimizer'] += value.numel() * value.element_size() / 1024 / 1024
    
    # Other allocated memory
    memory['other'] = memory['allocated'] - (memory['weights'] + 
                                           memory['gradients'] + 
                                           memory['optimizer'])
    
    # CUDA overhead
    memory['cuda_overhead'] = memory['nvidia_smi'] - memory['allocated']
    
    return memory

def print_memory_stats(gpu0_mem, gpu1_mem=None):
    """Print memory statistics for one or two GPUs"""
    print("\nGPU 0 Memory Usage:")
    print(f"  Weights:        {gpu0_mem['weights']:.1f} MB")
    print(f"  Gradients:      {gpu0_mem['gradients']:.1f} MB")
    print(f"  Optimizer:      {gpu0_mem['optimizer']:.1f} MB")
    print(f"  Other:          {gpu0_mem['other']:.1f} MB")
    print(f"  Allocated:      {gpu0_mem['allocated']:.1f} MB")
    print(f"  Cached:         {gpu0_mem['cached']:.1f} MB")
    print(f"  CUDA Overhead:  {gpu0_mem['cuda_overhead']:.1f} MB")
    print(f"  nvidia-smi:     {gpu0_mem['nvidia_smi']:.1f} MB")

    if gpu1_mem is not None:
        print("\nGPU 1 Memory Usage:")
        print(f"  Weights:        {gpu1_mem['weights']:.1f} MB")
        print(f"  Gradients:      {gpu1_mem['gradients']:.1f} MB")
        print(f"  Optimizer:      {gpu1_mem['optimizer']:.1f} MB")
        print(f"  Other:          {gpu1_mem['other']:.1f} MB")
        print(f"  Allocated:      {gpu1_mem['allocated']:.1f} MB")
        print(f"  Cached:         {gpu1_mem['cached']:.1f} MB")
        print(f"  CUDA Overhead:  {gpu1_mem['cuda_overhead']:.1f} MB")
        print(f"  nvidia-smi:     {gpu1_mem['nvidia_smi']:.1f} MB")
    print("-" * 50)

def plot_memory_usage(stages, gpu0_mems, gpu1_mems=None, base_filename='memory_usage', 
                     include_cuda_overhead=True, plot_both_gpus=True):
    """
    Plot memory usage for GPUs showing different memory components.
    
    Args:
        stages: List of stage names (x-axis labels)
        gpu0_mems: Memory measurements for GPU0 at each stage
        gpu1_mems: Memory measurements for GPU1 at each stage (optional)
        base_filename: Base name for the output file (e.g., 'single_gpu' or 'pipeline')
        include_cuda_overhead: Whether to show CUDA overhead in plot
        plot_both_gpus: Whether to plot two GPUs or just one
    """
    plt.switch_backend('Agg')
    
    # Define what memory components to show
    if include_cuda_overhead:
        memory_types = ['weights', 'gradients', 'optimizer', 'other', 'cuda_overhead']
        labels = ['Weights', 'Gradients', 'Optimizer', 'Other', 'CUDA Overhead']
        filename = f'{base_filename}_with_overhead.png'
    else:
        memory_types = ['weights', 'gradients', 'optimizer', 'other']
        labels = ['Weights', 'Gradients', 'Optimizer', 'Other']
        filename = f'{base_filename}_without_overhead.png'
    
    # Create figure with appropriate number of subplots
    if plot_both_gpus and gpu1_mems is not None:
        fig, (gpu0_plot, gpu1_plot) = plt.subplots(2, 1, figsize=(15, 10))
    else:
        fig, gpu0_plot = plt.subplots(1, 1, figsize=(15, 5))
    
    # Prepare x-axis
    x_positions = range(len(stages))
    
    # Plot GPU0
    bottom = np.zeros(len(stages))
    for memory_type, label in zip(memory_types, labels):
        memory_values = [mem[memory_type] for mem in gpu0_mems]
        gpu0_plot.bar(x_positions, memory_values, bottom=bottom, label=label)
        bottom += memory_values
    
    gpu0_plot.set_title('Single GPU Memory Usage' if not plot_both_gpus or gpu1_mems is None else 
                       'GPU 0 (Convolution Layers) Memory Usage')
    gpu0_plot.set_ylabel('Memory (MB)')
    gpu0_plot.set_xticks(x_positions)
    gpu0_plot.set_xticklabels(stages, rotation=45)
    gpu0_plot.legend()
    
    # Plot GPU1 if needed
    if plot_both_gpus and gpu1_mems is not None:
        bottom = np.zeros(len(stages))
        for memory_type, label in zip(memory_types, labels):
            memory_values = [mem[memory_type] for mem in gpu1_mems]
            gpu1_plot.bar(x_positions, memory_values, bottom=bottom, label=label)
            bottom += memory_values
        
        gpu1_plot.set_title('GPU 1 (Fully Connected Layers) Memory Usage')
        gpu1_plot.set_ylabel('Memory (MB)')
        gpu1_plot.set_xticks(x_positions)
        gpu1_plot.set_xticklabels(stages, rotation=45)
        gpu1_plot.legend()
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

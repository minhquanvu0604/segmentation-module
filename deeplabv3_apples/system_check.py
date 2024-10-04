import os
import torch

def check_system():
    """ 
    Check system capabilities and resources.
    
    Returns:
        dict: Information about the system's hardware and software environment.
    """
    system_info = {}

    # Check for GPU availability
    if torch.cuda.is_available():
        system_info['gpu_available'] = True
        system_info['gpu_name'] = torch.cuda.get_device_name(0)
        system_info['gpu_count'] = torch.cuda.device_count()
        system_info['gpu_memory'] = f"{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB"
        system_info['current_gpu'] = torch.cuda.current_device()
    else:
        system_info['gpu_available'] = False
    
    # Check number of CPU cores available
    system_info['cpu_count'] = os.cpu_count()

    # Check PyTorch version
    system_info['torch_version'] = torch.__version__

    # Check the amount of RAM available (requires psutil)
    try:
        import psutil
        ram_info = psutil.virtual_memory()
        system_info['total_ram'] = f"{ram_info.total / (1024 ** 3):.2f} GB"
        system_info['available_ram'] = f"{ram_info.available / (1024 ** 3):.2f} GB"
    except ImportError:
        system_info['psutil_installed'] = False
        print("Consider installing `psutil` for detailed RAM information: `pip install psutil`")
    
    return system_info

def print_system_info():
    """ Print the system's information in a readable format. """
    system_info = check_system()
    
    print("==== System Information ====")
    print(f"GPU Available: {system_info['gpu_available']}")
    if system_info['gpu_available']:
        print(f"GPU Name: {system_info['gpu_name']}")
        print(f"GPU Count: {system_info['gpu_count']}")
        print(f"GPU Memory: {system_info['gpu_memory']}")
        print(f"Current GPU Device ID: {system_info['current_gpu']}")
    
    print(f"CPU Cores: {system_info['cpu_count']}")
    print(f"PyTorch Version: {system_info['torch_version']}")
    
    if 'total_ram' in system_info:
        print(f"Total RAM: {system_info['total_ram']}")
        print(f"Available RAM: {system_info['available_ram']}")
    else:
        print("psutil is not installed, RAM information not available.")


def check_batch_size(device):
    # if device.type == 'cuda':
    #     total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # Get total GPU memory in GB
    #     if total_memory < 8:  # Example threshold
    #         config['batch_size'] = min(config['batch_size'], 8)
    #     txt_logger.info(f"Adjusted batch size: {config['batch_size']} based on available GPU memory")
    pass


if __name__ == "__main__":
    print_system_info()

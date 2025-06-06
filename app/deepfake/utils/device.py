import torch

def get_providers_from_device(device):
    providers = ['CPUExecutionProvider']
    if device == 'cuda':
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    elif device == 'coreml' or device == 'mps':
        providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
    elif device == 'rocm':
        providers = ['ROCMExecutionProvider', 'CPUExecutionProvider']
    return providers


def check_device(device):
    if device not in ["cuda", "mps", "rocm"]:
        if device == "coreml":
            device = "mps"
        else:
            print(f'deivce ({device}) unsupport, fallback to CPU')
            device = 'cpu'
    if device == 'cuda':
        if not torch.cuda.is_available() :
            device = 'cpu'
    if device == 'mps':
        if not torch.backends.mps.is_available():
            device = 'cpu'
    return device
import torch
print(f'torch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'cuDNN: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "N/A"}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(i)
        print(f'  {i}: {torch.cuda.get_device_name(i)} (sm_{cap[0]}{cap[1]})')

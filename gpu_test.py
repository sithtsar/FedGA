#!/usr/bin/env python3
"""
GPU utilization test for FL-GA implementation
"""

import torch
import time
import psutil
import GPUtil

def test_gpu_availability():
    """Test basic GPU availability and info"""
    print("=== GPU Availability Test ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    print()

def test_model_gpu_usage():
    """Test model forward/backward pass GPU usage"""
    print("=== Model GPU Usage Test ===")

    # Create model
    from src.fl_ga.model import create_model
    model = create_model()

    if torch.cuda.is_available():
        device = torch.device('cuda')
        model = model.to(device)

        # Test forward pass
        batch_size = 1024  # Larger batch for better GPU utilization
        x = torch.randn(batch_size, 1, 28, 28).to(device)
        model.eval()

        # Warm up
        with torch.no_grad():
            _ = model(x)

        # Time forward pass
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = model(x)
        torch.cuda.synchronize()
        end = time.time()

        print(".3f")
        print(".1f")

        # Test backward pass (training)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        y = torch.randint(0, 10, (batch_size,)).to(device)

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        end = time.time()

        print(".3f")
        print(".1f")
    else:
        print("CUDA not available - running on CPU")
        device = torch.device('cpu')
        model = model.to(device)

        x = torch.randn(32, 1, 28, 28).to(device)
        start = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = model(x)
        end = time.time()
        print(".3f")
    print()

def test_fl_bottlenecks():
    """Analyze potential FL bottlenecks"""
    print("=== FL Bottleneck Analysis ===")

    # Test data loading speed
    from src.fl_ga.data_loader import load_mnist
    print("Loading MNIST data...")
    start = time.time()
    client_datasets, test_dataset = load_mnist(num_clients=10, alpha=0.5)
    end = time.time()
    print(".2f")
    print(f"Client datasets: {len(client_datasets)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Average client dataset size: {sum(len(ds) for ds in client_datasets) / len(client_datasets):.0f}")
    print()

    # Test single client training speed
    from src.fl_ga.fl_base import train_local
    from src.fl_ga.model import create_model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing single client training on {device}")

    model = create_model()
    start = time.time()
    trained_model = train_local(model, client_datasets[0], epochs=1, device=device.type)
    end = time.time()
    print(".2f")
    print()

def monitor_system_resources():
    """Monitor system resources during a short test"""
    print("=== System Resource Monitoring ===")

    try:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            print(f"GPU {gpu.id}: {gpu.name}")
            print(".1f")
            print(".1f")
            print(".1f")
    except:
        print("GPUtil not available for detailed GPU monitoring")

    # CPU info
    print(f"CPU cores: {psutil.cpu_count()}")
    print(".1f")
    print()

if __name__ == "__main__":
    test_gpu_availability()
    test_model_gpu_usage()
    test_fl_bottlenecks()
    monitor_system_resources()

    print("=== Recommendations ===")
    print("1. FL is inherently sequential - GPU utilization will be bursty")
    print("2. Consider larger batch sizes (128-256) for better GPU utilization")
    print("3. Model is lightweight - consider deeper networks for research")
    print("4. Colab free tier has limited GPU performance")
    print("5. Monitor with 'nvidia-smi' or 'watch -n 1 nvidia-smi' during training")
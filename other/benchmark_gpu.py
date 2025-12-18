#!/usr/bin/env python3
"""
Comprehensive GPU benchmark for YOLO inference.
"""
import torch
from ultralytics import YOLO
import time

print("=" * 60)
print("YOLO GPU Benchmark")
print("=" * 60)

# GPU Info
print("\nGPU Information:")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("   No GPU detected!")
    exit(1)

# Load model once
print("\nLoading YOLO model...")
model = YOLO("yolov8n.pt")

# Warm-up run (to load model into GPU memory)
print("\nWarm-up run...")
_ = model.predict(source="examples/example_1.jpg", device=0, verbose=False)

# GPU Benchmark
print("\nBenchmark: 5 runs on GPU")
gpu_times = []
for i in range(5):
    start = time.time()
    results = model.predict(source="examples/example_1.jpg", device=0, verbose=False)
    elapsed = time.time() - start
    gpu_times.append(elapsed)
    print(f"   Run {i+1}: {elapsed:.4f}s")

avg_gpu = sum(gpu_times) / len(gpu_times)
print(f"   Average GPU Time: {avg_gpu:.4f}s")
print(f"   GPU Memory Used: {torch.cuda.memory_allocated(0) / 1e9:.4f} GB")

# CPU Benchmark
print("\nBenchmark: 5 runs on CPU")
cpu_times = []
for i in range(5):
    start = time.time()
    results = model.predict(source="examples/example_1.jpg", device='cpu', verbose=False)
    elapsed = time.time() - start
    cpu_times.append(elapsed)
    print(f"   Run {i+1}: {elapsed:.4f}s")

avg_cpu = sum(cpu_times) / len(cpu_times)
print(f"   Average CPU Time: {avg_cpu:.4f}s")

# Results
print("\n" + "=" * 60)
speedup = avg_cpu / avg_gpu
print(f"Results: GPU is {speedup:.2f}x faster than CPU")
print(f"GPU: {avg_gpu:.4f}s | CPU: {avg_cpu:.4f}s")
print("=" * 60)

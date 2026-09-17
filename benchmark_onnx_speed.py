"""
Benchmarks PyTorch vs. ONNX FP32 vs. ONNX INT8 inference speed on the same
input, so any claim about ONNX providing a latency benefit is backed by a
real measurement rather than an assumption. Run this AFTER
export_onnx_model.py.

Usage:
    python -m benchmark_onnx_speed
"""

import time
import os
import sys
import torch
import numpy as np

_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if os.path.isdir(_SRC_DIR) and _SRC_DIR not in sys.path:
    sys.path.append(_SRC_DIR)

from model import SkatingLSTMAutoencoder
from onnx_inference import ONNXFatigueDetector

WINDOW_SIZE = 30
N_FEATURES = 6
N_RUNS = 200
BATCH_SIZE = 8


def load_pytorch_model():
    model = SkatingLSTMAutoencoder(seq_len=WINDOW_SIZE, n_features=N_FEATURES, embedding_dim=64, num_phases=3)
    try:
        checkpoint = torch.load("skating_degradation_model.pth", map_location="cpu")
        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint.get("state_dict", checkpoint))
        else:
            model = checkpoint
    except Exception:
        pass
    model.eval()
    return model


def benchmark_pytorch(model, dummy_input, n_runs=N_RUNS):
    with torch.no_grad():
        for _ in range(10):  # warmup
            model(dummy_input)

        start = time.perf_counter()
        for _ in range(n_runs):
            model(dummy_input)
        elapsed = time.perf_counter() - start

    return elapsed / n_runs


def benchmark_onnx(detector, dummy_input_np, n_runs=N_RUNS):
    for _ in range(10):  # warmup
        detector.predict(dummy_input_np)

    start = time.perf_counter()
    for _ in range(n_runs):
        detector.predict(dummy_input_np)
    elapsed = time.perf_counter() - start

    return elapsed / n_runs


def main():
    dummy_input_torch = torch.randn(BATCH_SIZE, WINDOW_SIZE, N_FEATURES, dtype=torch.float32)
    dummy_input_np = dummy_input_torch.numpy()

    print(f"Benchmarking with batch_size={BATCH_SIZE}, {N_RUNS} runs each (after 10 warmup runs)\n")

    print("Loading PyTorch model...")
    pytorch_model = load_pytorch_model()
    pytorch_time = benchmark_pytorch(pytorch_model, dummy_input_torch)
    print(f"  PyTorch mean inference time: {pytorch_time * 1000:.3f} ms")

    print("\nLoading ONNX FP32 model...")
    onnx_fp32_detector = ONNXFatigueDetector(use_int8=False)
    onnx_fp32_time = benchmark_onnx(onnx_fp32_detector, dummy_input_np)
    print(f"  ONNX FP32 mean inference time: {onnx_fp32_time * 1000:.3f} ms")

    print("\nLoading ONNX INT8 model...")
    onnx_int8_detector = ONNXFatigueDetector(use_int8=True)
    onnx_int8_time = benchmark_onnx(onnx_int8_detector, dummy_input_np)
    print(f"  ONNX INT8 mean inference time: {onnx_int8_time * 1000:.3f} ms")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    fp32_speedup = pytorch_time / onnx_fp32_time
    int8_speedup = pytorch_time / onnx_int8_time
    print(f"ONNX FP32 vs PyTorch: {fp32_speedup:.2f}x {'faster' if fp32_speedup > 1 else 'slower'}")
    print(f"ONNX INT8 vs PyTorch: {int8_speedup:.2f}x {'faster' if int8_speedup > 1 else 'slower'}")
    print("\nUse these exact numbers (not estimates) if documenting an ONNX latency claim.")
    print("If INT8 is NOT faster than PyTorch on this hardware/batch size, say so honestly --")
    print("quantization benefits are hardware- and workload-dependent, not guaranteed.")


if __name__ == "__main__":
    main()

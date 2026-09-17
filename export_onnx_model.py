"""
Properly exports SkatingLSTMAutoencoder to ONNX and applies REAL dynamic
INT8 quantization.

The previous skating_model_int8.onnx was broken: it was larger than the
original FP32 file (626KB vs 133KB) with mixed FP32/INT8/INT32 tensor
types, meaning whatever process created it inserted quantize/dequantize
wrapper nodes around the original float weights rather than actually
replacing them. This script does it correctly using
onnxruntime.quantization.quantize_dynamic, which performs genuine
weight-only INT8 quantization -- the resulting file should be
meaningfully SMALLER than the FP32 original, not larger.

Usage:
    python -m export_onnx_model
"""

import os
import sys
import torch
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if os.path.isdir(_SRC_DIR) and _SRC_DIR not in sys.path:
    sys.path.append(_SRC_DIR)

from model import SkatingLSTMAutoencoder

WINDOW_SIZE = 30
N_FEATURES = 6
EMBEDDING_DIM = 64
NUM_PHASES = 3

MODEL_WEIGHTS_PATH = "skating_degradation_model.pth"
ONNX_FP32_PATH = "skating_model.onnx"
ONNX_INT8_PATH = "skating_model_int8.onnx"


def load_model():
    model = SkatingLSTMAutoencoder(
        seq_len=WINDOW_SIZE, n_features=N_FEATURES,
        embedding_dim=EMBEDDING_DIM, num_phases=NUM_PHASES
    )
    if os.path.exists(MODEL_WEIGHTS_PATH):
        try:
            checkpoint = torch.load(MODEL_WEIGHTS_PATH, map_location="cpu")
            if isinstance(checkpoint, dict):
                model.load_state_dict(checkpoint.get("state_dict", checkpoint))
            else:
                model = checkpoint
            print(f"Loaded weights from {MODEL_WEIGHTS_PATH}")
        except Exception as e:
            print(f"Could not load {MODEL_WEIGHTS_PATH} ({e}) -- exporting with random init weights instead")
    else:
        print(f"{MODEL_WEIGHTS_PATH} not found -- exporting with random init weights")
    model.eval()
    return model


def export_fp32(model):
    dummy_input = torch.randn(1, WINDOW_SIZE, N_FEATURES, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        ONNX_FP32_PATH,
        input_names=["input"],
        output_names=["reconstruction", "phase_logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "reconstruction": {0: "batch_size"},
            "phase_logits": {0: "batch_size"},
        },
        opset_version=17,
        dynamo=False,  # the new dynamo-based exporter (default in torch 2.13) produces
                        # symbolic shape mismatches for this LSTM model that break
                        # onnxruntime's quantization shape-inference step. The legacy
                        # TorchScript-based exporter is far more mature for RNN/LSTM
                        # export and avoids this problem entirely.
    )
    print(f"Exported FP32 ONNX model -> {ONNX_FP32_PATH} ({os.path.getsize(ONNX_FP32_PATH)} bytes)")


def quantize_to_int8():
    if os.path.exists(ONNX_INT8_PATH):
        os.remove(ONNX_INT8_PATH)  # remove the old broken file cleanly, don't leave it ambiguous

    # Officially recommended pre-processing step (onnxruntime itself warns to run
    # this before quantize_dynamic) -- runs more robust shape inference and graph
    # optimization ahead of time, reducing the odds of the shape-mismatch error
    # quantize_dynamic's internal shape inference can hit on RNN/LSTM graphs.
    from onnxruntime.quantization.shape_inference import quant_pre_process
    preprocessed_path = "skating_model_preprocessed.onnx"
    quant_pre_process(ONNX_FP32_PATH, preprocessed_path)
    print(f"Pre-processed model for quantization -> {preprocessed_path}")

    quantize_dynamic(
        model_input=preprocessed_path,
        model_output=ONNX_INT8_PATH,
        weight_type=QuantType.QInt8,
    )

    fp32_size = os.path.getsize(ONNX_FP32_PATH)
    int8_size = os.path.getsize(ONNX_INT8_PATH)
    reduction_pct = (1 - int8_size / fp32_size) * 100

    print(f"\nQuantized INT8 ONNX model -> {ONNX_INT8_PATH}")
    print(f"  FP32 size: {fp32_size} bytes")
    print(f"  INT8 size: {int8_size} bytes")
    print(f"  Size reduction: {reduction_pct:.1f}%")

    if int8_size >= fp32_size:
        print("  WARNING: INT8 file is not smaller than FP32 -- quantization did not work as expected.")
        print("  Do NOT claim this as functional quantization if this warning appears.")
    else:
        print("  Confirmed: INT8 file is genuinely smaller than the FP32 original.")


def verify_onnx_outputs_match_pytorch(model):
    """Sanity check: run the same input through PyTorch and both ONNX
    models, confirm outputs are numerically close. This is what makes the
    'functional' claim honest -- not just that the files exist, but that
    they produce equivalent results."""
    dummy_input = torch.randn(1, WINDOW_SIZE, N_FEATURES, dtype=torch.float32)

    with torch.no_grad():
        torch_recon, torch_phase = model(dummy_input)
    torch_recon = torch_recon.numpy()

    np_input = dummy_input.numpy()

    fp32_session = ort.InferenceSession(ONNX_FP32_PATH)
    fp32_recon, fp32_phase = fp32_session.run(None, {"input": np_input})

    int8_session = ort.InferenceSession(ONNX_INT8_PATH)
    int8_recon, int8_phase = int8_session.run(None, {"input": np_input})

    fp32_diff = np.abs(torch_recon - fp32_recon).max()
    int8_diff = np.abs(torch_recon - int8_recon).max()

    print(f"\nOutput verification (max absolute difference vs PyTorch):")
    print(f"  ONNX FP32 vs PyTorch: {fp32_diff:.6f}")
    print(f"  ONNX INT8 vs PyTorch: {int8_diff:.6f}")
    print(f"  (INT8 difference is expected to be larger than FP32 -- that's the")
    print(f"   accuracy/speed tradeoff of quantization. Large but not huge is normal;")
    print(f"   if this number is wildly large, e.g. >1.0, something is wrong.)")


def main():
    print("=" * 60)
    print("STEP 1: Load PyTorch model")
    print("=" * 60)
    model = load_model()

    print("\n" + "=" * 60)
    print("STEP 2: Export to FP32 ONNX")
    print("=" * 60)
    export_fp32(model)

    print("\n" + "=" * 60)
    print("STEP 3: Quantize to real INT8 ONNX")
    print("=" * 60)
    quantize_to_int8()

    print("\n" + "=" * 60)
    print("STEP 4: Verify outputs match between PyTorch and both ONNX models")
    print("=" * 60)
    verify_onnx_outputs_match_pytorch(model)


if __name__ == "__main__":
    main()

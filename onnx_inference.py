"""
Real ONNX inference wrapper. This is what makes the ONNX capability
actually FUNCTIONAL rather than an unused import -- pipeline_engine.py
(or the dashboard) should call run_onnx_inference() instead of the raw
PyTorch model when edge acceleration is enabled.

Usage:
    from onnx_inference import ONNXFatigueDetector
    detector = ONNXFatigueDetector(use_int8=True)
    reconstruction, phase_logits = detector.predict(window_batch)
"""

import os
import numpy as np
import onnxruntime as ort

ONNX_FP32_PATH = "skating_model.onnx"
ONNX_INT8_PATH = "skating_model_int8.onnx"


class ONNXFatigueDetector:
    """Loads a real onnxruntime.InferenceSession and runs actual inference
    through it -- the piece that was missing before (onnxruntime was
    imported in the dashboard but this class, or anything like it, never
    existed and was never called)."""

    def __init__(self, use_int8=True):
        model_path = ONNX_INT8_PATH if use_int8 else ONNX_FP32_PATH
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"{model_path} not found. Run `python -m export_onnx_model` first "
                f"to generate it."
            )
        self.session = ort.InferenceSession(model_path)
        self.model_path = model_path
        self.use_int8 = use_int8

    def predict(self, window_batch):
        """window_batch: numpy array of shape (batch_size, seq_len, n_features).
        Returns (reconstruction, phase_logits) as numpy arrays."""
        window_batch = np.asarray(window_batch, dtype=np.float32)
        reconstruction, phase_logits = self.session.run(
            None, {"input": window_batch}
        )
        return reconstruction, phase_logits

    def compute_reconstruction_loss(self, window_batch):
        """Convenience method matching the loss computation pattern used
        elsewhere in the pipeline (mean squared error per window)."""
        reconstruction, _ = self.predict(window_batch)
        window_batch = np.asarray(window_batch, dtype=np.float32)
        per_window_loss = np.mean((reconstruction - window_batch) ** 2, axis=(1, 2))
        return per_window_loss


if __name__ == "__main__":
    # Quick smoke test -- confirms the session actually loads and runs,
    # not just that the file exists.
    detector = ONNXFatigueDetector(use_int8=True)
    dummy_batch = np.random.randn(4, 30, 6).astype(np.float32)
    losses = detector.compute_reconstruction_loss(dummy_batch)
    print(f"ONNX INT8 inference ran successfully on model: {detector.model_path}")
    print(f"Per-window reconstruction losses for 4 random windows: {losses}")

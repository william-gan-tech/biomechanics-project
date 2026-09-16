import os
import torch
import onnx
from onnxruntime.quantization import quantize_static, CalibrationDataReader
from src.model import SkatingLSTMAutoencoder

class SkatingCalibrationDataReader(CalibrationDataReader):
    def __init__(self, calibration_data):
        self.data = calibration_data
        self.enum_data = iter(self.data)

    def get_next(self):
        try:
            return next(self.enum_data)
        except StopIteration:
            return None

def export_and_quantize():
    model_path = "skating_model.onnx"
    quant_path = "skating_model_int8.onnx"

    # 1. Load trained PyTorch model and export to FP32 ONNX
    model = SkatingLSTMAutoencoder(seq_len=30, n_features=6, embedding_dim=64)
    try:
        model.load_state_dict(torch.load("skating_degradation_model.pth"))
    except FileNotFoundError:
        print("Trained weights not found. Run src/train.py first.")
        return
    
    model.eval()
    dummy_input = torch.randn(1, 30, 6, dtype=torch.float32)
    
    torch.onnx.export(
        model,
        dummy_input,
        model_path,
        export_params=True,
        opset_version=14,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )

    # 2. Provide sample calibration data matching 6 features
    calib_data = [{"input": torch.randn(1, 30, 6, dtype=torch.float32).numpy()} for _ in range(10)]
    reader = SkatingCalibrationDataReader(calib_data)

    # 3. Run Static INT8 Quantization
    quantize_static(
        model_input=model_path,
        model_output=quant_path,
        weight_type=onnx.TensorProto.INT8,
        calibration_data_reader=reader
    )
    print(f"Successfully generated quantized edge model: {quant_path}")

if __name__ == "__main__":
    export_and_quantize()
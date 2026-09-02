import os
import torch
import onnx
from onnxruntime.quantization import quantize_static, CalibrationDataReader

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

    # 1. Dummy export to FP32 ONNX (replace with your actual PyTorch model export)
    # dummy_input = torch.randn(1, 30, 34)
    # torch.onnx.export(model, dummy_input, model_path, input_names=["input"], output_names=["output"])

    if not os.path.exists(model_path):
        print(f"Base model {model_path} not found. Export your PyTorch model first.")
        return

    # 2. Provide sample calibration data (batch of dummy tensors matching input shape)
    calib_data = [{"input": torch.randn(1, 30, 34).numpy()} for _ in range(10)]
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
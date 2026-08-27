import torch
import torch.onnx
from model import SkatingLSTMAutoencoder

def export_model_to_onnx(model_path="skating_degradation_model.pth", output_onnx_path="skating_model.onnx"):
    # 1. Initialize the model architecture (matching your pipeline settings)
    window_size = 30
    n_features = 4
    model = SkatingLSTMAutoencoder(seq_len=window_size, n_features=n_features, embedding_dim=64)

    # 2. Load trained weights
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    if isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint.get('state_dict', checkpoint))
    else:
        model = checkpoint
    
    model.eval()
    print("Model loaded successfully for export.")

    # 3. Create dummy input tensor matching your window shape [batch_size, seq_len, n_features]
    dummy_input = torch.randn(1, window_size, n_features, dtype=torch.float32)

    # 4. Export to ONNX format
    torch.onnx.export(
        model,                          # model being run
        dummy_input,                    # model input (or a tuple for multiple inputs)
        output_onnx_path,               # where to save the model
        export_params=True,             # store the trained parameter weights inside the model file
        opset_version=14,               # the ONNX version to export the model to
        do_constant_folding=True,       # whether to execute constant folding for optimization
        input_names=['input_sequence'], # the model's input names
        output_names=['reconstruction'],# the model's output names
        dynamic_axes={
            'input_sequence': {0: 'batch_size'},    # variable length batch size
            'reconstruction': {0: 'batch_size'}
        }
    )
    
    print(f"Success! Model exported to {output_onnx_path}")

if __name__ == "__main__":
    # Since we are running from root, just use the direct filename or point to root correctly
    export_model_to_onnx(model_path="skating_degradation_model.pth", output_onnx_path="skating_model.onnx")
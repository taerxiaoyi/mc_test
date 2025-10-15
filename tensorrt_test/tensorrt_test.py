import torch
import tensorrt as trt
import onnx

def convert_pt_to_engine(pt_path, onnx_path, engine_path, input_shape=(1, 3, 224, 224)):
    """完整的.pt到.engine转换流程"""
    
    # 1. 加载PyTorch模型
    print("步骤1: 加载PyTorch模型...")
    model = torch.load(pt_path, map_location='cpu')
    if isinstance(model, dict):
        model = model['model']  # 如果保存的是state_dict
    model.eval()
    
    # 2. 导出ONNX
    print("步骤2: 导出ONNX模型...")
    dummy_input = torch.randn(*input_shape)
    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=12
    )
    
    # 3. 验证ONNX
    print("步骤3: 验证ONNX模型...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    # 4. 构建TensorRT引擎
    print("步骤4: 构建TensorRT引擎...")
    build_engine(onnx_path, engine_path)
    
    print(f"转换完成! Engine文件保存在: {engine_path}")

# 使用示例
convert_pt_to_engine('teleopTask.pt', 'teleopTask.onnx', 'teleopTask.engine')

import tensorrt as trt

def check_engine_version(engine_file_path):
    # 创建日志记录器和运行时
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    
    # 尝试加载引擎文件
    try:
        with open(engine_file_path, "rb") as f:
            engine_data = f.read()
        engine = runtime.deserialize_cuda_engine(engine_data)
        # 如果加载成功，说明版本匹配，可以打印当前版本
        print(f"[成功] 引擎文件加载成功！当前TensorRT运行时版本: {trt.__version__}")
        
    except RuntimeError as e:
        error_msg = str(e)
        print(f"[错误信息] {error_msg}")
        # 从错误信息中提取期望的版本号
        # 例如："Error Code 6: ... expecting library version 10.13.0.35 got 8.6.1.6"
        if "expecting library version" in error_msg:
            # 提取"expecting library version"后面的数字版本号
            import re
            match = re.search(r"expecting library version\s+([\d.]+)", error_msg)
            if match:
                expected_version = match.group(1)
                print(f"\n>>> 引擎文件编译时使用的TensorRT版本为: {expected_version} <<<")
            else:
                print("无法从错误信息中解析出版本号。")
        else:
            print("错误信息格式不符，无法解析版本。")

if __name__ == "__main__":
    # 替换为你的.engine文件路径
    engine_path = "/home/mc/Desktop/policy_deploy-multi/resources/policies/g1/teleopTask.engine"
    check_engine_version(engine_path)

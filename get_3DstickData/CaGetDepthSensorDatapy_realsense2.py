import pyrealsense2 as rs
import numpy as np
import cv2

# 初始化RealSense管道
pipeline = rs.pipeline()
config = rs.config()

# 启用深度流
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# 开始流
profile = pipeline.start(config)

# 获取深度传感器的深度尺度
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(f"深度尺度: {depth_scale}")

# 创建对齐对象（将深度图对齐到彩色图，如果需要）
align_to = rs.stream.color
align = rs.align(align_to)

try:
    while True:
        # 等待帧
        frames = pipeline.wait_for_frames()
        
        # 对齐深度帧到彩色帧（可选）
        # aligned_frames = align.process(frames)
        # depth_frame = aligned_frames.get_depth_frame()
        
        depth_frame = frames.get_depth_frame()
        
        if not depth_frame:
            continue
        
        # 转换为numpy数组
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # 将深度图转换为米
        depth_image_meters = depth_image * depth_scale
        
        # 创建可视化深度图（归一化到0-255）
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), 
            cv2.COLORMAP_JET
        )
        
        # 显示深度图
        cv2.imshow('Depth Image', depth_colormap)
        
        # 打印统计信息
        print(f"\r深度范围: {np.min(depth_image_meters):.2f}-{np.max(depth_image_meters):.2f}m "
              f"中心深度: {depth_image_meters[240, 320]:.2f}m", end='')
        
        # 按q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    
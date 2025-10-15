"""
realtime_mediapipe_open3d.py
实时从摄像头读取人体3D关键点，并用Open3D绘制流畅的3D骨架。
按 Ctrl+C 或 关闭窗口退出。
"""
"""
使用 MediaPipe BlazePose (实时 3D 关键点) + Open3D 做 流畅的实时 3D 骨架可视化。脚本特点：
从摄像头或视频读取帧（默认摄像头 0）
用 MediaPipe 获取 pose_world_landmarks（真实/归一化的 3D 点）
用 Open3D 的 LineSet + PointCloud 实时更新渲染，流畅且可旋转/缩放
可选保存每帧 3D 坐标到 .npy 文件（开关控制）
"""

import cv2
import numpy as np
import mediapipe as mp
import open3d as o3d
import time
import argparse
from collections import deque

# ---------------------------
# 参数（可从命令行调整）
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--source", type=str, default="4", help="0 for camera, or path to video file")
parser.add_argument("--save", action="store_true", help="是否保存每帧的3D坐标到.npy（大文件）")
parser.add_argument("--max-save-frames", type=int, default=10000, help="保存帧数上限（避免无限大文件）")
args = parser.parse_args()

# ---------------------------
# 初始化 MediaPipe Pose
# ---------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=2,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

# ---------------------------
# 打开视频源
# ---------------------------
source = int(args.source) if args.source.isnumeric() else args.source
cap = cv2.VideoCapture(source)
if not cap.isOpened():
    raise RuntimeError(f"无法打开视频源: {args.source}")

# ---------------------------
# 构建 Open3D 可视化窗口、几何体
# ---------------------------
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Real-time 3D Pose - Open3D", width=960, height=720, visible=True)

# 初始点（MediaPipe BlazePose 有 33 个 keypoints）
NUM_KEYPOINTS = 33

# 初始 points 全为 0
points = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float64)

# 转换 MediaPipe 的 connections（set of tuples）为 line 索引列表
connections = list(mp_pose.POSE_CONNECTIONS)  # 每项为 (i1, i2)
lines = [[int(a), int(b)] for (a, b) in connections]

# 创建 LineSet（骨架）
line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(points),
    lines=o3d.utility.Vector2iVector(lines)
)
line_set.colors = o3d.utility.Vector3dVector([[0.1, 0.4, 0.8] for _ in lines])  # 线颜色

# 创建点云用于显示关节点（更容易控制点大小）
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector([[1.0, 0.2, 0.2] for _ in range(NUM_KEYPOINTS)])

vis.add_geometry(line_set)
vis.add_geometry(pcd)

# 调整渲染选项（背景、点大小等）
opt = vis.get_render_option()
opt.background_color = np.asarray([0.95, 0.95, 0.95])
opt.point_size = 8.0
opt.line_width = 3.0
opt.mesh_show_back_face = True

# 添加坐标轴辅助（可选）
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0,0,0])
vis.add_geometry(axis)

# ---------------------------
# 用于保存帧数据（可选）
# ---------------------------
saved_frames = []
save_count = 0

# 为防止绘图闪烁和控制帧率：使用小队列存储历史点（平滑可视化）
SMOOTH_WINDOW = 3
history = deque(maxlen=SMOOTH_WINDOW)

# ---------------------------
# 实时主循环
# ---------------------------
try:
    last_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("视频结束或无法读取帧。")
            break

        # 可缩放或跳过帧以提高速度：如需更低延迟可在此处下采样
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        # 显示原始图像（窗口）——非必须，但方便调试
        cv2.imshow("Camera (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if results.pose_world_landmarks:   # 推荐使用 world_landmarks（具有3D坐标）
            lm = results.pose_world_landmarks.landmark
            pts = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float64)
            for i, p in enumerate(lm):
                # MediaPipe world_landmarks 已经是相对真实尺度的三维坐标（单位大致为米）
                pts[i, 0] = p.x
                pts[i, 1] = p.y
                pts[i, 2] = p.z

            # 可选：做一个简单的平滑（平均历史帧）以更流畅
            history.append(pts)
            avg_pts = np.mean(np.stack(list(history), axis=0), axis=0)

            # 更新 Open3D 几何体
            line_set.points = o3d.utility.Vector3dVector(avg_pts)
            pcd.points = o3d.utility.Vector3dVector(avg_pts)

            # 必须显式更新几何体
            vis.update_geometry(line_set)
            vis.update_geometry(pcd)

            # 渲染事件循环
            vis.poll_events()
            vis.update_renderer()

            # 保存（可选）
            if args.save and save_count < args.max_save_frames:
                saved_frames.append(avg_pts.copy())
                save_count += 1

            # 打印 FPS（可选）
            now = time.time()
            fps = 1.0 / (now - last_time) if now != last_time else 0.0
            last_time = now
            # 在终端输出每秒帧率（会比较频繁，可注释）
            # print(f"FPS: {fps:.1f}", end="\r")

        else:
            # 没检测到人体时仍然要保活渲染循环（否则窗口会卡死）
            vis.poll_events()
            vis.update_renderer()

except KeyboardInterrupt:
    print("\n用户中断（KeyboardInterrupt）——退出程序。")

finally:
    cap.release()
    cv2.destroyAllWindows()
    vis.destroy_window()
    pose.close()

    # 保存.npy（若启用）
    if args.save and saved_frames:
        arr = np.stack(saved_frames, axis=0)  # shape: (num_frames, 33, 3)
        out_path = "saved_pose_world.npy"
        np.save(out_path, arr)
        print(f"已保存 {arr.shape[0]} 帧的3D坐标到: {out_path}")

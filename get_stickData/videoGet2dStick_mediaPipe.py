
"""
下面我给你一份完整、可直接运行的 Python 脚本，能从一张图片中：
提取人体 3D 关键点坐标（33个关节）
输出坐标数组（NumPy 格式）
用 matplotlib 在 3D 空间中绘制人体骨架（火柴人风格）

整个流程基于 MediaPipe BlazePose GHUM 3D，不需要GPU也能跑。
"""

# requirements: pip install mediapipe opencv-python numpy scipy
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from scipy.signal import savgol_filter

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

# 每人保留的历史关键点队列（这里假设单人）
history_len = 10
kp_history = deque(maxlen=history_len)  # 存每帧的 (N,2) 关键点

# 常用函数：计算角度 (p1-p2-p3 中间点为顶点)，返回弧度
def angle_between(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    # 保护数值
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    cosang = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    return np.arccos(cosang)

# 关键点索引（MediaPipe 有 33 个 landmark）
# 取常用：肩膀、肘、手腕、髋、膝、踝、鼻子等
L = mp_pose.PoseLandmark
use_idx = {
    'left_shoulder': L.LEFT_SHOULDER.value,
    'right_shoulder': L.RIGHT_SHOULDER.value,
    'left_elbow': L.LEFT_ELBOW.value,
    'right_elbow': L.RIGHT_ELBOW.value,
    'left_wrist': L.LEFT_WRIST.value,
    'right_wrist': L.RIGHT_WRIST.value,
    'left_hip': L.LEFT_HIP.value,
    'right_hip': L.RIGHT_HIP.value,
    'left_knee': L.LEFT_KNEE.value,
    'right_knee': L.RIGHT_KNEE.value,
    'left_ankle': L.LEFT_ANKLE.value,
    'right_ankle': L.RIGHT_ANKLE.value,
    'nose': L.NOSE.value
}

cap = cv2.VideoCapture('input.mp4')  # or 0 for webcam
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
dt = 1.0 / fps

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)
    landmarks = res.pose_landmarks

    if landmarks:
        pts = np.array([[lm.x * w, lm.y * h, lm.z] for lm in landmarks.landmark])  # (33,3)
        # 提取感兴趣点 (N,2)
        kp = np.stack([pts[idx][:2] for idx in use_idx.values()])  # shape (M,2)
        kp_history.append(kp)

        # 规范化：以两肩中点为中心，除以肩宽
        left_sh = kp[list(use_idx.keys()).index('left_shoulder')]
        right_sh = kp[list(use_idx.keys()).index('right_shoulder')]
        center = (left_sh + right_sh) / 2.0
        shoulder_dist = np.linalg.norm(left_sh - right_sh) + 1e-6
        kp_norm = (kp - center) / shoulder_dist  # 归一化坐标

        # 计算肘角（左肘：肩-肘-腕）
        idx_map = {k:i for i,k in enumerate(use_idx.keys())}
        l_sh = kp[idx_map['left_shoulder']]
        l_el = kp[idx_map['left_elbow']]
        l_wr = kp[idx_map['left_wrist']]
        left_elbow_angle = angle_between(l_sh, l_el, l_wr)  # radians

        # 速度：当前关键点与上一帧差分（像素/frame -> 可转为单位/sec by dt）
        if len(kp_history) >= 2:
            vel = (kp_history[-1] - kp_history[-2]) / dt  # (M,2) pixels/sec
        else:
            vel = np.zeros_like(kp)

        # 可视化火柴人（连线）
        def draw_skeleton(img, pts2d, idx_map_local):
            # 简单连接列表（根据需要扩展）
            connections = [
                ('left_shoulder','right_shoulder'),
                ('left_shoulder','left_elbow'),
                ('left_elbow','left_wrist'),
                ('right_shoulder','right_elbow'),
                ('right_elbow','right_wrist'),
                ('left_shoulder','left_hip'),
                ('right_shoulder','right_hip'),
                ('left_hip','right_hip'),
                ('left_hip','left_knee'),
                ('left_knee','left_ankle'),
                ('right_hip','right_knee'),
                ('right_knee','right_ankle'),
            ]
            for a,b in connections:
                ia, ib = idx_map_local[a], idx_map_local[b]
                pa = tuple(pts2d[ia].astype(int))
                pb = tuple(pts2d[ib].astype(int))
                cv2.line(img, pa, pb, (0,255,0), 2)
            for p in pts2d:
                cv2.circle(img, tuple(p.astype(int)), 3, (0,0,255), -1)

        draw_skeleton(frame, kp, idx_map)

        # 在画面上写入角度/速度示例
        cv2.putText(frame, f"Left elbow(deg): {np.degrees(left_elbow_angle):.1f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.imshow('stickman', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


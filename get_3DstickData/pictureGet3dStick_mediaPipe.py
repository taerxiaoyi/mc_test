import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =========================
# 1️⃣ 初始化 MediaPipe Pose
# =========================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False)

# =========================
# 2️⃣ 读取图片并推理
# =========================
image_path = "person5.jpg"  # 👈 替换为你的图片路径
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"未找到图片: {image_path}")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = pose.process(image_rgb)

# =========================
# 3️⃣ 获取 3D 关键点坐标
# =========================
if not results.pose_world_landmarks:
    print("未检测到人体！")
    exit()

landmarks = results.pose_world_landmarks.landmark
joints_3d = np.array([[l.x, l.y, l.z] for l in landmarks])

print("3D关键点数组 shape:", joints_3d.shape)
print(joints_3d[:5])  # 打印前5个关节坐标

# =========================
# 4️⃣ 3D 可视化骨架
# =========================
connections = mp_pose.POSE_CONNECTIONS  # MediaPipe 提供的关节连接关系

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('3D Human Pose (MediaPipe BlazePose)')

# 绘制关节点
ax.scatter(joints_3d[:, 0], joints_3d[:, 1], joints_3d[:, 2], c='r', s=30)

# 绘制连接线
for connection in connections:
    i1, i2 = connection
    x = [joints_3d[i1, 0], joints_3d[i2, 0]]
    y = [joints_3d[i1, 1], joints_3d[i2, 1]]
    z = [joints_3d[i1, 2], joints_3d[i2, 2]]
    ax.plot(x, y, z, c='b', linewidth=2)

# 调整视角与比例
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(elev=15, azim=-70)
ax.set_box_aspect([1, 1, 1])
plt.tight_layout()
plt.show()

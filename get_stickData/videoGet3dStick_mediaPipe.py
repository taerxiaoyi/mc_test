
"""
我们可以基于 MediaPipe BlazePose GHUM 3D 来实现这一整套实时流程。
它支持：

单摄像头实时人体关键点检测；

33个3D关键点；

z为相对深度坐标；

可在 CPU 实时运行。
"""

"""
五、性能建议

若你电脑性能有限，可把 model_complexity=1；

若想更快可视化，可以使用 Open3D 替代 matplotlib；

若想存数据（比如训练用），可以在每帧保存 joints_3d 数组。
"""
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ===============================
# 1️⃣ 初始化 MediaPipe Pose 模型
# ===============================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=2,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

# ===============================
# 2️⃣ 打开视频流（摄像头或视频文件）
# ===============================
cap = cv2.VideoCapture(4)  # 摄像头：0；视频文件："video.mp4"
# cap = cv2.VideoCapture("input.mp4")  # 摄像头：0；视频文件："video.mp4"

# ===============================
# 3️⃣ 设置 Matplotlib 实时3D绘图
# ===============================
plt.ion()
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')

def draw_skeleton_3d(ax, joints_3d, connections):
    ax.cla()  # 清空当前帧
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=15, azim=-70)
    ax.set_title("Real-Time 3D Pose (MediaPipe BlazePose)")

    # 绘制关节
    ax.scatter(joints_3d[:, 0], joints_3d[:, 1], joints_3d[:, 2], c='r', s=25)

    # 绘制骨架连接线
    for connection in connections:
        i1, i2 = connection
        x = [joints_3d[i1, 0], joints_3d[i2, 0]]
        y = [joints_3d[i1, 1], joints_3d[i2, 1]]
        z = [joints_3d[i1, 2], joints_3d[i2, 2]]
        ax.plot(x, y, z, c='b', linewidth=2)

# ===============================
# 4️⃣ 实时循环：读取帧 → 检测 → 绘图
# ===============================
connections = mp_pose.POSE_CONNECTIONS

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # BGR → RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    # 可选：在2D窗口中显示原图
    cv2.imshow('Camera', frame)

    if results.pose_world_landmarks:
        landmarks = results.pose_world_landmarks.landmark
        joints_3d = np.array([[l.x, l.y, l.z] for l in landmarks])

        # 绘制3D骨架
        draw_skeleton_3d(ax, joints_3d, connections)
        plt.pause(0.001)

    # 按 q 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===============================
# 5️⃣ 结束
# ===============================
cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()

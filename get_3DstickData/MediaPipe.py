import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)  # 单张图片模式
image = cv2.imread("person.jpg")
results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

if results.pose_landmarks:
    # 获取3D关键点
    landmarks = results.pose_world_landmarks.landmark
    joints_3d = np.array([[l.x, l.y, l.z] for l in landmarks])
    print(joints_3d.shape)  # (33, 3)

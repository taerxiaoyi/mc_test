# -*- coding: utf-8 -*-

import time 
import numpy as np
# from isaacgym.torch_utils import *
import torch

import sys, os 
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import mujoco, mujoco.viewer

import copy
import json
import threading
from datetime import datetime
import imageio
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R


class E1SimEnv():

    _max_fps = 800
    target_state = {
        'dof_pos': np.zeros(29, dtype=np.float32),
        'dof_vel': np.zeros(29, dtype=np.float32),
        'root_rot': np.array([1, 0, 0, 0], dtype=np.float32),
        'root_xyz': np.zeros(3, dtype=np.float32),
        'root_ang_vel': np.zeros(3, dtype=np.float32)}
    state = None
    _root_ang_vel_is_local = False


    _lock = threading.Lock()
    _lock_read = threading.Lock()
    _lock_write = threading.Lock()

    def __init__(self, config, fps, kps, kds):

        self.config = config
        E1SimEnv.fps = fps
        E1SimEnv.kps = kps
        E1SimEnv.kds = kds
        E1SimEnv.DOF_LOWER = np.array(self.config.dof_lower).copy()
        E1SimEnv.DOF_UPPER = np.array(self.config.dof_upper).copy()
        E1SimEnv.EFFORT_LIMIT = np.array(self.config.effort_limit).copy()
        
        step_thread = threading.Thread(target=E1SimEnv.step_thread)
        step_thread.start()
        time.sleep(1)
    
    @staticmethod
    def read_state():
        with E1SimEnv._lock_read:
            state = E1SimEnv.state.copy()
        return state
    
    @staticmethod
    def step_thread():
        max_fps = E1SimEnv._max_fps
        model = mujoco.MjModel.from_xml_path("resources/robots/o1/o1_fixed.xml")
        # model = mujoco.MjModel.from_xml_path("test/resources/robots/g1/g1_fixed.xml")

        model.opt.timestep = 1./max_fps
        data = mujoco.MjData(model)
        mujoco.mj_step(model, data)
        viewer = mujoco.viewer.launch_passive(model, data) #mujoco_viewer.MujocoViewer(model, data)
        written_frames = 0

        kp = np.array(E1SimEnv.kps).copy()
        kd = np.array(E1SimEnv.kds).copy()
        torque_limit = np.array(E1SimEnv.EFFORT_LIMIT).copy() 

        with E1SimEnv._lock:
            E1SimEnv.target_state['dof_pos'] = data.qpos.astype(np.float32).copy()[7:].copy()
            E1SimEnv.target_state['dof_vel'] = data.qvel.astype(np.float32).copy()[6:].copy()
            E1SimEnv.target_state['root_rot'] = data.qpos.astype(np.float32).copy()[3:7].copy()
            E1SimEnv.target_state['root_xyz'] = data.qpos.astype(np.float32).copy()[0:3].copy()
            E1SimEnv.target_state['root_ang_vel'] = data.qvel.astype(np.float32).copy()[3:6].copy()

        from loop_rate_limiters import RateLimiter
        rate_limiter = RateLimiter(frequency=max_fps, warn=False)
        last_render_time = time.time()
        last_record_time = time.time()

        while True:

            with E1SimEnv._lock:
                target_dof_pos = E1SimEnv.target_state['dof_pos'].copy()
                target_dof_vel = E1SimEnv.target_state['dof_vel'].copy()
                root_rot = E1SimEnv.target_state['root_rot'].copy()
                root_xyz = E1SimEnv.target_state['root_xyz'].copy()
                root_ang_vel = E1SimEnv.target_state['root_ang_vel'].copy()

            data.qpos[7:] = target_dof_pos.copy()
            data.qvel[6:] = 0.
            data.qpos[3:7] = root_rot#[[1, 2, 3, 0]]
            data.qpos[0:3] = root_xyz
            data.qvel[3:6] = 0

            # dof_pos = data.qpos.astype(np.float32).copy()[7:]
            # dof_vel = data.qvel.astype(np.float32).copy()[6:]

            # torque = (action - dof_pos) * kp - dof_vel * kd
            # torque = np.clip(torque, -torque_limit, torque_limit)
            # data.ctrl = torque.copy()
            mujoco.mj_step(data.model, data)
            # mujoco.mj_forward(data.model, data)

            # time.sleep(1./max_fps)
            rate_limiter.sleep()

            dof_pos = data.qpos.astype(np.float32).copy()[7:]
            dof_vel = data.qvel.astype(np.float32).copy()[6:]
            dof_torques = data.ctrl.astype(np.float32).copy()
          
            root_rot = data.qpos.astype(np.float32).copy()[3:7]#[[1, 2, 3, 0]]
            root_xyz = data.qpos.astype(np.float32).copy()[0:3]
            root_ang_vel = data.qvel.astype(np.float32).copy()[3:6]
            with E1SimEnv._lock_read:
                E1SimEnv.state = {"time": time.time(),
                    "dof_pos": dof_pos, "dof_vel": dof_vel, "dof_torques": dof_torques,
                    "root_rot": root_rot, "root_xyz": root_xyz, "root_ang_vel": root_ang_vel}


            if time.time() - last_render_time > 1./50:
                viewer.sync()#viewer.render()
                last_render_time = time.time()
        viewer.close()

    def step(self, target_state):

        with E1SimEnv._lock:
            for key in target_state:
                    if key in E1SimEnv.target_state:
                        E1SimEnv.target_state[key] = target_state[key].copy()

        return

    def move_to_default_pos(self):
        # move time 2s
        total_time = 2
        control_dt = self.config.control_dt  # 使用仿真的控制周期
        num_step = int(total_time / control_dt)
        default_joint_pos = np.array(self.config.default_joint_pos).copy()
        
        # 获取当前状态
        current_state = self.read_state()
        init_dof_pos = current_state['dof_pos'].copy()
        
        # Smoothly move to default position using interpolation
        for step in range(num_step):
            alpha = step / num_step
            
            # 线性插值计算目标位置
            target_dof_pos = init_dof_pos * (1 - alpha) + default_joint_pos * alpha
            target_dof_vel = np.zeros_like(target_dof_pos)  # 目标速度为0
            
            # 构建目标状态
            target_state = {
                'dof_pos': target_dof_pos,
                'dof_vel': target_dof_vel,
                'root_rot': current_state['root_rot'].copy(),  # 保持当前旋转
                'root_xyz': current_state['root_xyz'].copy(),  # 保持当前位置
                'root_ang_vel': np.zeros(3, dtype=np.float32)  # 角速度为0
            }
            
            # 发送控制命令
            self.step(target_state)
            time.sleep(control_dt)
        return

if __name__ == "__main__":
    print("hello world")



    
    


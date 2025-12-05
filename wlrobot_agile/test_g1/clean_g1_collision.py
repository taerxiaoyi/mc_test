import os
import sys
import time
import argparse
import pdb
import os.path as osp

import glob

sys.path.append(os.getcwd())

from motion.poselib.poselib.skeleton.skeleton3d import SkeletonTree
import torch

import numpy as np
import math
from copy import deepcopy
from collections import defaultdict
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as sRot
import joblib


def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Adds one capsule to an mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1  # increment ngeom
    # initialise a new capsule, add it to the scene using mjv_makeConnector
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom - 1],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.zeros(9), rgba.astype(np.float32))
    mujoco.mjv_makeConnector(scene.geoms[scene.ngeom - 1],
                             mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                             point1[0], point1[1], point1[2],
                             point2[0], point2[1], point2[2])


def key_call_back(keycode):
    pass


def check_self_collision(model, data):
    for i in range(data.ncon):
        contact = data.contact[i]
        # geom1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
        # geom2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)

        body1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[contact.geom1])
        body2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[contact.geom2])

        penetration_depth = -contact.dist
        forces = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, forces)

        # Compute the magnitude of the normal force
        normal_force_magnitude = np.linalg.norm(forces[:3])
        if penetration_depth > 1e-6:
            print(f"Collision detected between {body1_name} and {body2_name}, penetration depth:{penetration_depth}, force:{normal_force_magnitude:.2f}")

            return True
    return False


def main():

    # folder --> folder_clean
    folder = f"data/figure_data/t02aM"
    folder_clean = f"data/figure_data/t02aM_clean"
    os.makedirs(folder_clean, exist_ok=True)

    # # folder/* --> new_folder/*
    # folder = f"/home/hx/code/EMAN/sample_data/A/*"
    # folder_clean = f"/home/hx/code/EMAN/sample_data/A_clean/*"

    motions = glob.glob(os.path.join(folder, "*.pkl"), recursive=True)
    current_id, max_id = 0, len(motions)
    cleaned_data, bad_data = 0, 0

    humanoid_xml = 'resources/robots/g1/g1_29dof_1.xml'
    mj_model = mujoco.MjModel.from_xml_path(humanoid_xml)
    mj_model.geom_margin = 0.00001
    mj_model.opt.iterations = 100
    # mj_model.opt.timestep = 0.001

    mj_data = mujoco.MjData(mj_model)
    
    current_id = -1
    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_call_back) as viewer:
        while viewer.is_running() and current_id < max_id:
            current_id += 1

            if current_id >= max_id:
                print("所有动作已处理完成")
                break

            curr_motion_key = motions[current_id].split("/")[-1].replace(".pkl", "")
            curr_motion = joblib.load(motions[current_id])#[curr_motion_key]
            assert curr_motion['fps'] == 30, f"{curr_motion_key}: {curr_motion['fps']}"

            _folder_clean = motions[current_id].replace(folder.replace("/*", "/"), folder_clean.replace("/*", "/"))
            _folder_clean = "/".join(_folder_clean.split("/")[:-1])
            os.makedirs(_folder_clean, exist_ok=True)

            _len = curr_motion['dof_from_pose_aa'].shape[0]
            i = 0
            for i in range(_len):
                mj_data.qpos[:3] = curr_motion['root_trans_offset'][i]
                mj_data.qpos[3:7] = curr_motion['root_rot'][i][[3, 0, 1, 2]]
                mj_data.qpos[7:] = curr_motion['dof_from_pose_aa'][i]

                mujoco.mj_forward(mj_model, mj_data)
                # viewer.sync()

                self_collision = check_self_collision(mj_model, mj_data)
                if self_collision:
                    print(f"检测到碰撞，丢弃动作: {curr_motion_key}")
                    bad_data += 1
                    
                    total_processed = cleaned_data + bad_data
                    print(f"坏数据率: {bad_data/total_processed:.2%}, 已处理: {total_processed}/{max_id}")
                    break

                if i == _len - 1:
                    print(f"保存干净动作: {curr_motion_key}")
                    cleaned_data += 1

                    total_processed = cleaned_data + bad_data
                    print(f"坏数据率: {bad_data/total_processed:.2%}, 已处理: {total_processed}/{max_id}")
                    _unique = "" # str(int(time.time()))
                    with open(os.path.join(_folder_clean, curr_motion_key + _unique + ".pkl"), 'wb') as f:
                        joblib.dump(curr_motion, f)
            print("")

    # 处理完成后的总结
    print(f"\n处理完成!")
    print(f"总动作数: {max_id}")
    print(f"干净动作: {cleaned_data}")
    print(f"有碰撞动作: {bad_data}")
    if cleaned_data + bad_data > 0:
        print(f"最终坏数据率: {bad_data/(cleaned_data + bad_data):.2%}")


if __name__ == "__main__":
    main()
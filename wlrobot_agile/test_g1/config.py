import numpy as np
import yaml

class Config:
    def __init__(self, file_path) -> None:
        with open(file_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

            self.control_dt = config["control_dt"]
            self.kps = config["kps"]
            self.kds = config["kds"]
            self.joint_names = config["joint_names"]
            self.default_joint_pos = np.array(config["default_joint_pos"], dtype=np.float32)
            self.cont_pos1 = np.array(config["cont_pos1"], dtype=np.float32)
            self.num_joints = config["num_joints"]
            self.dof_lower = config["dof_lower"]
            self.dof_upper = config["dof_upper"] 
            self.hold_pos = config["hold_pos"]
            self.motion_speeds = config["motion_speeds"]
            self.motion_lower_bounds = config["motion_lower_bounds"]
            self.motion_upper_bounds = config["motion_upper_bounds"]
            self.effort_limit = config["effort_limit"]
            self.policy_path = config.get("policy_path", None)
            self.xml = config.get("xml", "e1.xml")
            self.msg_type = config.get("msg_type", "agile")  # "agile" 
            self.imu_type = config.get("imu_type", "pelvis")  # "torso" or "pelvis"
            self.lowcmd_topic = config.get("lowcmd_topic", "rt/lowcmd")
            self.lowstate_topic = config.get("lowstate_topic", "rt/lowstate")
            
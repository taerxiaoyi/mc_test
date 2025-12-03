from legged_gym import LEGGED_GYM_ROOT_DIR
import numpy as np
import time
# import torch

from westlake_sdkpy.core.channel import ChannelPublisher, ChannelFactoryInitialize
from westlake_sdkpy.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from westlake_sdkpy.idl.default import agile_msg_dds__LowCmd_, agile_msg_dds__LowState_

from westlake_sdkpy.idl.agile.msg.dds_ import LowCmd_ as LowCmdAG
from westlake_sdkpy.idl.agile.msg.dds_ import LowState_ as LowStateAG

from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_agile, MotorMode
from common.remote_controller import RemoteController, KeyMap
from config import Config


class Controller:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.remote_controller = RemoteController()

        # Initialize the policy network
        # self.policy = torch.jit.load(config.policy_path)
        # Initializing process variables
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.target_dof_pos = config.default_angles.copy()
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        self.cmd = np.array([0.0, 0, 0])
        self.counter = 0


        if config.msg_type == "agile":
            self.low_cmd = agile_msg_dds__LowCmd_()
            self.low_state = agile_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdAG)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateAG)
            self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

        else:
            raise ValueError("Invalid msg_type")

        # wait for the subscriber to receive data
        self.wait_for_low_state()

        if config.msg_type == "agile":
            # Initialize the command msg
            init_cmd_agile(self.low_cmd, self.mode_machine_, self.mode_pr_)


    def LowStateHgHandler(self, msg: LowStateAG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)


    def send_cmd(self, cmd: LowCmdAG):
        self.lowcmd_publisher_.Write(cmd)


    def wait_for_low_state(self):
        while self.low_state.sequences == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")


    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.START] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)


    def move_to_default_pos(self):
        print("Moving to default pos.")
        # move time 2s
        total_time = 2
        num_step = int(total_time / self.config.control_dt)
        
        dof_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx
        kps = self.config.kps + self.config.arm_waist_kps
        kds = self.config.kds + self.config.arm_waist_kds
        default_pos = np.concatenate((self.config.default_angles, self.config.arm_waist_target), axis=0)
        dof_size = len(dof_idx)
        
        # record the current pos
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q
        
        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            for j in range(dof_size):
                motor_idx = dof_idx[j]
                target_pos = default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].dq = 0
                self.low_cmd.motor_cmd[motor_idx].kp = kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def default_pos_state(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        while self.remote_controller.button[KeyMap.A] != 1:
            for i in range(len(self.config.leg_joint2motor_idx)):
                motor_idx = self.config.leg_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[i]
                self.low_cmd.motor_cmd[motor_idx].dq = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            for i in range(len(self.config.arm_waist_joint2motor_idx)):
                motor_idx = self.config.arm_waist_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_waist_target[i]
                self.low_cmd.motor_cmd[motor_idx].dq = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def run(self):
        self.counter += 1
        # Get the current joint position and velocity
        # 打印腿部电机
        print("==== 腿部电机 ====")
        for i in range(len(self.config.leg_joint2motor_idx)):
            motor_idx = self.config.leg_joint2motor_idx[i]
            pos = self.low_state.motor_state[motor_idx].q
            vel = self.low_state.motor_state[motor_idx].dq
            print(f"腿部关节{i} (ID={motor_idx}): 位置={pos:.3f}, 速度={vel:.3f}")

        # 打印手部/腰部电机
        print("\n==== 手部/腰部电机 ====")
        for i in range(len(self.config.arm_waist_joint2motor_idx)):
            motor_idx = self.config.arm_waist_joint2motor_idx[i]
            pos = self.low_state.motor_state[motor_idx].q
            vel = self.low_state.motor_state[motor_idx].dq
            print(f"手部关节{i} (ID={motor_idx}): 位置={pos:.3f}, 速度={vel:.3f}")
                
            # imu_state quaternion: w, x, y, z
            quat = self.low_state.imu_state.quaternion
            ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)

        print("\nIMU数据:")
        print(f"四元数(w,x,y,z): {quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}")
        print(f"角速度(rad/s): [X={ang_vel[0][0]:.3f}, Y={ang_vel[0][1]:.3f}, Z={ang_vel[0][2]:.3f}]")

        # send the command
        self.send_cmd(self.low_cmd)

        time.sleep(self.config.control_dt)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    parser.add_argument("config", type=str, help="config file name in the configs folder", default="agile.yaml")
    args = parser.parse_args()

    # Load config
    config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_real/configs/{args.config}"
    config = Config(config_path)

    # Initialize DDS communication
    ChannelFactoryInitialize(0, args.net)

    controller = Controller(config)

    # Enter the zero torque state, press the start key to continue executing
    controller.zero_torque_state()

    # Move to the default position
    controller.move_to_default_pos()

    # Enter the default position state, press the A key to continue executing
    controller.default_pos_state()

    while True:
        try:
            controller.run()
            # Press the select key to exit
            if controller.remote_controller.button[KeyMap.BACK] == 1:
                break
        except KeyboardInterrupt:
            break
    # Enter the damping state
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")

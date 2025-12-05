from typing import Dict, List, Any, Optional
import numpy as np
import time

from westlake_sdkpy.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from westlake_sdkpy.idl.default import agile_msg_dds__LowCmd_, agile_msg_dds__LowState_
from westlake_sdkpy.idl.agile.msg.dds_ import LowCmd_ as LowCmdAG
from westlake_sdkpy.idl.agile.msg.dds_ import LowState_ as LowStateAG

# from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
# from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
# from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
# from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
# from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
# from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
# from unitree_sdk2py.utils.crc import CRC

from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_agile, MotorMode
from common.remote_controller import RemoteController, KeyMap
from config import Config

# from eman.base.rotation_helper import transform_imu_data
from scipy.spatial.transform import Rotation as R
def transform_imu_data(waist_yaw, waist_yaw_omega, imu_quat, imu_omega):
    RzWaist = R.from_euler("z", waist_yaw).as_matrix()
    R_torso = R.from_quat([imu_quat[1], imu_quat[2], imu_quat[3], imu_quat[0]]).as_matrix()
    R_pelvis = np.dot(R_torso, RzWaist.T)
    w = np.dot(RzWaist, imu_omega[0]) - np.array([0, 0, waist_yaw_omega])
    return R.from_matrix(R_pelvis).as_quat()[[3, 0, 1, 2]], w

class E1RealEnv:
    """
    Robot control environment class for interacting with a real robot.
    
    Uses DDS communication to receive robot states and send control commands, supporting multiple control modes.
    """
    
    def __init__(self, config: Config) -> None:
        """
        Initialize the robot control environment.
        
        Args:
            config: Configuration object containing all necessary parameter settings
        """
        self.config = config
        self.remote_controller = RemoteController()

        # Initialize policy network
        # self.policy = torch.jit.load(config.policy_path)
        
        # Initialize process variables
        self.action = np.zeros(config.num_joints, dtype=np.float32)
        self.target_dof_pos = config.default_joint_pos.copy()
        self.cmd = np.array([0.0, 0, 0])
        self.counter = 0
        self.low_state = None  # Initialize low_state to None

        _lowcmd_topic = "rt/lowcmd"
        _lowstate_topic = "rt/lowstate"

        if config.msg_type == "agile":
            self.low_cmd = agile_msg_dds__LowCmd_()
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0

            # Initialize DDS publishers and subscribers
            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdAG)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateAG)
            self.lowstate_subscriber.Init(self.low_state_handler, 10)

        # elif config.msg_type == "hg":
        #     self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        #     self.low_state = unitree_hg_msg_dds__LowState_()

        #     self.mode_pr_ = MotorMode.PR
        #     self.mode_machine_ = 0
        #     self.motor_mode = 1

        #     self.lowcmd_publisher_ = ChannelPublisher(_lowcmd_topic, LowCmdHG)
        #     self.lowcmd_publisher_.Init()

        #     self.lowstate_subscriber = ChannelSubscriber(_lowstate_topic, LowStateHG)
        #     self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)
        else:
            raise ValueError(f"Unsupported msg_type: {config.msg_type}")

        # Wait to receive robot state
        self.wait_for_low_state()

        if config.msg_type == "agile":
            # Initialize command message
            init_cmd_agile(self.low_cmd, self.mode_machine_, self.mode_pr_)

    def low_state_handler(self, msg: LowStateAG) -> None:
        """
        Process received robot state messages.
        
        Args:
            msg: Received LowStateAG message
        """
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    # def LowStateHgHandler(self, msg: LowStateHG) -> None:
    #     """
    #     Process received robot state messages.
        
    #     Args:
    #         msg: Received LowStateAG message
    #     """
    #     self.low_state = msg
    #     self.mode_machine_ = self.low_state.mode_machine
    #     self.remote_controller.set(self.low_state.wireless_remote)

    # def send_cmd(self, cmd: LowStateHG) -> None:
    #     """
    #     Send control commands to the robot.
        
    #     Args:
    #         cmd: Control command to send
    #     """
    #     cmd.crc = CRC().Crc(cmd)
    #     self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self, timeout: float = 10.0) -> None:
        """
        Wait to receive robot state data.
        
        Args:
            timeout: Timeout duration in seconds
            
        Raises:
            TimeoutError: If no data is received within the timeout period
        """
        start_time = time.time()
        # while self.low_state is None or self.low_state.tick == 0:
        while self.low_state is None or self.low_state.sequences == 0:
            if time.time() - start_time > timeout:
                raise TimeoutError("Timeout waiting for robot state")
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self) -> None:
        """
        Enter zero-torque state, allowing all motors to free-wheel.
        
        Waits for the START button signal from the remote controller.
        """
        print("Entering zero-torque state.")
        print("Waiting for start signal...")
        
        while self.remote_controller.button[KeyMap.START] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
       
        # move time 2s
        total_time = 2
        num_step = int(total_time / self.config.control_dt)
        
        # dof_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx
        kps = self.config.kps
        kds = self.config.kds
        default_joint_pos = self.config.default_joint_pos
        dof_size = 29 #len(dof_idx)
        
        # record the current pos
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[i].pos
        
        # move to default pos
        # Smoothly move to default position
        for step in range(num_step):
            alpha = step / num_step
            for j in range(dof_size):
                motor_idx = j
                target_pos = default_joint_pos[j]
                
                # Calculate interpolation between current position and target position
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].dq = 0
                self.low_cmd.motor_cmd[motor_idx].kp = kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
                
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)
        
        # 等待一小段时间让电机稳定
        time.sleep(0.1)
        
        # 检查是否达到目标位置
        position_tolerance = 0.02
        all_reached = True
        unreached_joints = []
        
        for j in range(dof_size):
            current_pos = self.low_state.motor_state[j].q
            target_pos = default_joint_pos[j]
            position_error = abs(current_pos - target_pos)
            
            if position_error > position_tolerance:
                all_reached = False
                unreached_joints.append((j, current_pos, target_pos, position_error))
        
        # 输出结果
        if all_reached:
            print("所有关节均已到达默认位置！")
            return True
        else:
            print(f"警告：{len(unreached_joints)} 个关节未到达目标位置（容差：{position_tolerance} rad）")
            for joint_info in unreached_joints:
                j, current, target, error = joint_info
                print(f"  关节 {j}: 当前位置={current:.4f}, 目标位置={target:.4f}, 误差={error:.4f}")
            return False


    
    def default_pos_state(self) -> None:
        """
        Maintain default position state.
        
        Waits for the A button signal from the remote controller.
        """
        print("Entering default position state.")
        print("Waiting for A button signal...")
            
        while self.remote_controller.button[KeyMap.A] != 1:
            for i in range(self.config.num_joints):
                motor_idx = i
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_joint_pos[i]
                self.low_cmd.motor_cmd[motor_idx].dq = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def read_state(self) -> Dict[str, np.ndarray]:
        """
        Read current robot state.
        
        Returns:
            Dictionary containing various robot state information
        """
        if self.low_state is None:
            raise ValueError("No robot state data received")
            
        num_joints = self.config.num_joints
        joint_pos = np.zeros(num_joints, dtype=np.float32)
        joint_vel = np.zeros(num_joints, dtype=np.float32)
        joint_torque = np.zeros(num_joints, dtype=np.float32)
        temperatures = np.zeros(num_joints, dtype=np.float32)

        # Read joint states
        for i in range(num_joints):
            joint_pos[i] = self.low_state.motor_state[i].q
            joint_vel[i] = self.low_state.motor_state[i].dq
            joint_torque[i] = self.low_state.motor_state[i].tau_est
            # temperatures[i] = self.low_state.motor_state[i].temperature

        # Read IMU state
        quat = np.array(self.low_state.imu_state.quaternion, dtype=np.float32)  # w, x, y, z
        ang_vel = np.array(self.low_state.imu_state.gyroscope, dtype=np.float32)
        
        # If IMU type is pelvis, coordinate transformation is needed
        # if self.config.imu_type == "pelvis":
        #     waist_yaw = self.low_state.motor_state[12].pos
        #     waist_yaw_omega = self.low_state.motor_state[12].w
        #     quat, ang_vel = transform_imu_data(
        #         waist_yaw=waist_yaw,
        #         waist_yaw_omega=waist_yaw_omega,
        #         imu_quat=quat,
        #         imu_omega=ang_vel
        #     )

        return {
            "dof_pos": joint_pos,
            "dof_vel": joint_vel,
            "dof_torque": joint_torque,
            "root_rot": quat,
            "root_ang_vel": ang_vel,
            # "temperatures": temperatures
        }

    def step(self, step_action: Dict[str, np.ndarray],kps ,kds ) -> None:
        """
        Execute one control action step.
        
        Args:
            step_action: Dictionary containing control actions, must have an "action" key
        """
            
        action = step_action
        
        for i in range(self.config.num_joints):
            motor_idx = i
            self.low_cmd.motor_cmd[motor_idx].q = action[i]
            self.low_cmd.motor_cmd[motor_idx].dq = 0
            self.low_cmd.motor_cmd[motor_idx].kp = kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0
            
        self.send_cmd(self.low_cmd)

    def close(self) -> None:
        """Clean up resources and disconnect."""
        # Send zero-torque command to put robot in safe state
        create_zero_cmd(self.low_cmd)
        self.send_cmd(self.low_cmd)
        
        # Clean up DDS resources
        if hasattr(self, 'lowcmd_publisher_'):
            self.lowcmd_publisher_.Close()
        if hasattr(self, 'lowstate_subscriber'):
            self.lowstate_subscriber.Close()
            
        print("Robot connection closed.")


if __name__ == "__main__":
    # Run the connection test when this file is executed directly
    

   # Simple connection test

    print("Starting connection test...")
    
    # Create a simple configuration for testing

    
    try:
        # Initialize DDS
        ChannelFactoryInitialize(0)
        
        # Create environment instance
        config_path = "test/g1.yaml"
        config = Config(config_path)
        env = E1RealEnv(config)
        
        print("✓ Successfully connected to robot")
        
        # Test reading state
        state = env.read_state()
        print(f"✓ Successfully read robot state with {len(state['dof_pos'])} joints")
        print(f"  Joint positions: {state['dof_pos']}")
        
        # Test sending a simple command
        test_action = {"action": config.default_joint_pos.copy()}
        env.step(test_action)
        print("✓ Successfully sent control command")
        
        # Wait a bit to see if any errors occur
        time.sleep(1.0)
        
        # Clean up
        env.close()
        print("✓ Successfully closed connection")
 
    except Exception as e:
        print(f"✗ Connection test failed: {e}")


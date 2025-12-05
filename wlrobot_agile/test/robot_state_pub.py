import time
import numpy as np
from typing import Dict, Any

from westlake_sdkpy.core.channel import ChannelPublisher, ChannelFactoryInitialize
from westlake_sdkpy.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from westlake_sdkpy.idl.default import agile_msg_dds__LowCmd_
from westlake_sdkpy.idl.agile.msg.dds_ import LowCmd_

from westlake_sdkpy.idl.default import agile_msg_dds__LowState_
from westlake_sdkpy.idl.agile.msg.dds_ import LowState_
from westlake_sdkpy.utils.thread import RecurrentThread

# from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
# from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
# from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
# from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
# from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
# from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
# from unitree_sdk2py.utils.crc import CRC
# from unitree_sdk2py.utils.thread import RecurrentThread
from config import Config
from functools import partial


class RobotStateSimulator:
    """
    Simulates robot state feedback for testing purposes.
    Mimics the behavior of a real robot by publishing simulated state data.
    """
    
    def __init__(self, config=None):
        """
        Initialize the robot state simulator.
        
        Args:
            config: Configuration object with parameters for simulation
        """
        self.control_dt = 0.002  # 2ms control cycle
        # self.lowstate = unitree_hg_msg_dds__LowState_()
        self.lowstate = agile_msg_dds__LowState_()
        
        # Default configuration if none provided
        if config is None:
            self.config = type('Config', (), {})()
            self.config.num_joints = 29  
            self.config.default_joint_pos = np.array([0.0] * 29)
            self.config.arm_waist_joint2motor_idx = []
        else:
            self.config = config
            
        # Initialize state variables
        self.sequences = 0
        self.tick = 0
        self.mode_machine = 0
        self.wireless_remote = [0] * 40  # Simulate remote controller data
        
        # Initialize motor states
        # for i in range(self.config.num_joints):
        #     self.lowstate.motor_state[i].pos = self.config.default_joint_pos[i]
        #     self.lowstate.motor_state[i].w = 0.0  # velocity
        #     self.lowstate.motor_state[i].t = 0.0  # torque
        #     self.lowstate.motor_state[i].temperature = 40.0  # reasonable temperature
            
        # Initialize IMU state
        self.lowstate.imu_state.quaternion = [0.0, 0.0, 0.0,1.0]  # identity quaternion
        self.lowstate.imu_state.gyroscope = [0.0, 0.0, 0.0]
        self.lowstate.imu_state.accelerometer = [0.0, 0.0, 9.8]  # gravity
        
        # Command tracking
        self.last_cmd = None
        self.cmd_subscriber = None
        
    def init(self):
        """Initialize the simulator's publishers and subscribers."""
        # Create publisher for low state
        self.lowstate_publisher = ChannelPublisher("rt/lowstate", LowState_)
        self.lowstate_publisher.Init()
        
        # Create subscriber for low commands (to simulate response to commands)
        self.cmd_subscriber = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self.cmd_subscriber.Init(self.command_handler, 10)
        
        return True
        
    def start(self):
        """Start the simulation threads."""
        self.state_publish_thread = RecurrentThread(
            interval=self.control_dt, 
            target=self.publish_state,
            name="state_publisher"
        )
        self.state_publish_thread.Start()
        
        print("Robot state simulator started")
        return True
        
    def command_handler(self, cmd: LowCmd_):
        """
        Handle incoming commands and update simulation accordingly.
        
        Args:
            cmd: The received LowCmd_ command
        """
        self.last_cmd = cmd
        
        # Simulate response to position commands with some latency
        for i in range(min(self.config.num_joints, len(cmd.motor_cmd))):
            if hasattr(cmd.motor_cmd[i], 'pos'):
                # Simple simulation: move toward commanded position with some dynamics
                current_pos = self.lowstate.motor_state[i].q
                target_pos = cmd.motor_cmd[i].q
                
                # Simple first-order response
                k = 0.1  # response rate
                new_pos = current_pos + k * (target_pos - current_pos)
                
                self.lowstate.motor_state[i].q = new_pos
                
                # Simulate velocity based on position change
                self.lowstate.motor_state[i].dq = k * (target_pos - current_pos) / self.control_dt
                
    def publish_state(self):
        """Publish the current simulated state."""
        # self.tick += 1
        # self.lowstate.tick = self.tick
        self.sequences += 1
        self.lowstate.sequences = self.sequences
        self.lowstate.mode_machine = self.mode_machine
        self.lowstate.wireless_remote = self.wireless_remote
        
        # Add some small random noise to simulate real sensor data
        for i in range(self.config.num_joints):
            self.lowstate.motor_state[i].q += np.random.normal(0, 0.0005)
            self.lowstate.motor_state[i].dq += np.random.normal(0, 0.005)
            
        # Add small noise to IMU
        for i in range(3):
            self.lowstate.imu_state.gyroscope[i] += np.random.normal(0, 0.0005)
            self.lowstate.imu_state.accelerometer[i] += np.random.normal(0, 0.005)
            
        self.lowstate_publisher.Write(self.lowstate)
        
    def set_remote_button(self, button_index, value):
        """
        Simulate pressing a button on the remote controller.
        
        Args:
            button_index: Index of the button to simulate
            value: Value to set (0 or 1)
        """
        if 0 <= button_index < len(self.wireless_remote):
            self.wireless_remote[button_index] = value
            
    def set_joint_position(self, joint_index, position):
        """
        Directly set a joint position (for testing specific scenarios).
        
        Args:
            joint_index: Index of the joint to set
            position: Position value to set
        """
        if 0 <= joint_index < self.config.num_joints:
            self.lowstate.motor_state[joint_index].q = position
            
    def set_imu_orientation(self, quaternion):
        """
        Set the IMU orientation.
        
        Args:
            quaternion: List of 4 values [w, x, y, z] representing the orientation
        """
        if len(quaternion) == 4:
            self.lowstate.imu_state.quaternion = quaternion
            
    def stop(self):
        """Stop the simulator."""
        if hasattr(self, 'state_publish_thread'):
            self.state_publish_thread.Stop()
            
        print("Robot state simulator stopped")



if __name__ == '__main__':
    # Initialize DDS
    ChannelFactoryInitialize(0)
    
    config_path = "test/o1.yaml"
    config = Config(config_path)
    
    # Create and start the simulator
    simulator = RobotStateSimulator(config)
    simulator.init()
    simulator.start()
    while True:
            time.sleep(1)
    
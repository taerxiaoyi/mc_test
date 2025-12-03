import sys
import os
sys.path.append(os.getcwd())
from typing import Union, List, Dict, Optional
import numpy as np
import time
import threading
from collections import deque
import yaml
from loop_rate_limiters import RateLimiter
import math
from sim_robot import E1SimEnv
from real_robot import E1RealEnv
from base_controller import BaseController
from config import Config
import matplotlib.pyplot as plt
from westlake_sdkpy.core.channel import ChannelFactoryInitialize

ANIM_SEEK_LOWER = 1
ANIM_SEEK_UPPER = 2
ANIM_SEEK_DEFAULT = 3
ANIM_FINISHED = 4
ANIM_SEEK_SCALE = 5
ANIM_SINE_CUSTOM = 6

class Controller(BaseController):
    def __init__(self, config: Config, test_name: str, preview: bool = True, visualization: bool = False) -> None:
        super().__init__(config, test_name, preview, visualization)
        self.current_kp = None
        self.current_kd = None
        self.current_scale = None

    def run(self, joint_indices: Union[int, List[int]], step_count: int, 
            modes: Optional[Dict[int, int]] = None, current_speed_scale: float = None):
        """
        运行控制循环
        
        Args:
            joint_indices: 单个关节索引或关节索引列表
            step_count: 步数计数
            modes: 每个关节的运动模式字典 {joint_idx: anim_state}
            current_speed_scale: 当前速度比例
        """
        if isinstance(joint_indices, int):
            joint_indices = [joint_indices]
        
        if modes is None:
            # 使用每个关节的当前状态
            modes = {joint_idx: self.joint_states[joint_idx].anim_state for joint_idx in joint_indices}
        
        # 更新速度比例
        if current_speed_scale is not None:
            for joint_idx in joint_indices:
                self.joint_states[joint_idx].speed = current_speed_scale * min(
                    self.config.dof_upper[joint_idx] - self.config.dof_lower[joint_idx],
                    3.0 * math.pi
                )
        
        if self.preview:
            # 仿真模式
            sim_state = self.e1_sim.read_state()
            target_dof_pos, dof_deltas = self.cmd_joints(
                sim_state['dof_pos'].copy(), 
                target_joints=joint_indices,
                modes=modes
            )            
            sim_state['dof_pos'] = target_dof_pos.copy()
            self.e1_sim.step(sim_state.copy())
            
        else:
            # 真实机器人模式
            real_state = self.e1_real.read_state()
            target_dof_pos, dof_deltas = self.cmd_joints(
                real_state['dof_pos'].copy(), 
                target_joints=joint_indices,
                modes=modes
            )
            
            # 如果有上一个周期的数据，记录对齐的数据
            if hasattr(self, 'last_control_data') and self.last_control_data is not None:
                (last_joint_indices, last_target_positions, last_target_velocities, 
                last_compute_torques, last_kps, last_kds, last_speeds) = self.last_control_data
                
                # 当前读取的状态对应上一个周期命令的结果
                actual_positions = [real_state['dof_pos'][idx] for idx in last_joint_indices]
                actual_velocities = [real_state['dof_vel'][idx] for idx in last_joint_indices]
                actual_torques = [real_state['dof_torque'][idx] for idx in last_joint_indices]
                
                # 记录对齐的数据
                self.add_multi_joint_data(
                    last_joint_indices,
                    last_target_positions,      
                    actual_positions,           
                    last_target_velocities,     
                    actual_velocities,          
                    last_compute_torques,       
                    actual_torques,             
                    last_kps,                   
                    last_kds,                  
                    last_speeds                 
                )
            
            # 计算当前周期的控制参数
            target_positions = []
            target_velocities = []
            compute_torques = []
            kps = []
            kds = []
            speeds = []
            
            for joint_idx in joint_indices:
                position_error = target_dof_pos[joint_idx] - real_state['dof_pos'][joint_idx]
                target_velocity = dof_deltas[joint_idx] / self.config.control_dt
                
                compute_torque = self.compute_torque(position_error, real_state['dof_vel'][joint_idx], joint_idx)
                
                target_positions.append(target_dof_pos[joint_idx])
                target_velocities.append(target_velocity)
                compute_torques.append(compute_torque)
                kps.append(self.config.kps[joint_idx])
                kds.append(self.config.kds[joint_idx])
                speeds.append(self.joint_states[joint_idx].speed)
            
            # 保存当前周期的控制数据供下一个周期使用
            self.last_control_data = (
                joint_indices,
                target_positions,
                target_velocities,
                compute_torques,
                kps,
                kds,
                speeds
            )
            
            # 发送控制命令到真实机器人
            self.e1_real.step(target_dof_pos.copy(), kps=self.config.kps, kds=self.config.kds)
        
            # 同时更新仿真（用于可视化）
            self.e1_sim.step(real_state.copy())
                
        # 限速
        self.rate_limiter.sleep()
        
        # 记录控制时间（用于性能监控）
        self.log_control_time()
        
        # 更新可视化
        if self.visualization:
            self.update_visualization(step_count)


# 测试模式常量
TEST_RANDOM = 1      # 随机速度测试
TEST_STEP = 2        # 阶跃速度测试
TEST_SWEEP = 3       # 扫频测试
TEST_CUSTOM = 4      # 自定义测试序列


def generate_test_sequence(test_mode: int, steps_per_phase: int = 1000) -> List[float]:
    """
    生成测试速度序列
    
    Args:
        test_mode: 测试模式
        steps_per_phase: 每个阶段的步数
        
    Returns:
        速度比例序列
    """
    if test_mode == TEST_RANDOM:
        # 随机测试：每100步随机变化一次速度
        sequence = []
        for i in range(40):  # 生成40个随机速度段
            speed = np.random.uniform(0.01, 1.0)
            sequence.extend([speed] * 100)  # 每个速度持续100步
        return sequence
    
    elif test_mode == TEST_STEP:
        # 阶跃测试：低→中→高→低
        sequence = []
        for speed in [0.1, 0.3, 0.5, 0.7, 1.0, 0.7, 0.5, 0.3, 0.1]:
            sequence.extend([speed] * steps_per_phase)
        return sequence
    
    elif test_mode == TEST_SWEEP:
        # 扫频测试：速度从低到高再到低
        sequence = []
        # 上升阶段
        for i in range(20):
            speed = 0.1 + 0.045 * i  # 0.1, 0.145, 0.19, ..., 1.0
            sequence.extend([speed] * (steps_per_phase // 20))
        # 下降阶段
        for i in range(20):
            speed = 1.0 - 0.045 * i  # 1.0, 0.955, 0.91, ..., 0.1
            sequence.extend([speed] * (steps_per_phase // 20))
        return sequence
    
    elif test_mode == TEST_CUSTOM:
        # 自定义测试序列
        sequence = []
        # 低速测试
        sequence.extend([0.1] * steps_per_phase)
        # 中速测试
        sequence.extend([0.5] * steps_per_phase)
        # 高速测试
        sequence.extend([1.0] * steps_per_phase)
        # 超高速测试
        sequence.extend([2.0] * (steps_per_phase // 2))
        # 回低速测试
        sequence.extend([0.1] * steps_per_phase)
        return sequence
    
    else:
        # 默认使用阶跃测试
        sequence = []
        for speed in [0.1, 0.3, 0.5, 0.7, 1.0]:
            sequence.extend([speed] * steps_per_phase)
        return sequence


def main():
    import argparse

    parser = argparse.ArgumentParser(description="机器人关节速度变化测试")
    parser.add_argument("--config", type=str, help="configs文件夹中的配置文件名称", default="o1.yaml")
    parser.add_argument("--joint_idx", type=int, help="要控制的关节索引", default=2)
    parser.add_argument("--joints", type=str, help="要控制的多个关节索引，用逗号分隔，例如：0,1,2", default=None)
    parser.add_argument("--min_range", type=float, help="最小关节位置(弧度)", default=-2)
    parser.add_argument("--max_range", type=float, help="最大关节位置(弧度)", default=2)
    parser.add_argument("--preview", type=bool, default=True, help="在预览模式（仿真）下运行")
    parser.add_argument("--visualization", action="store_true", help="启用可视化")
    parser.add_argument("--test_mode", type=int, choices=[1, 2, 3, 4], 
                       help="测试模式: 1=随机, 2=阶跃, 3=扫频, 4=自定义", default=1)
    parser.add_argument("--steps_per_phase", type=int, help="每个速度阶段的步数", default=500)
    args = parser.parse_args()

    ChannelFactoryInitialize(0)

    # 加载配置
    config_path = f"test/{args.config}"
    config = Config(config_path)

    args.joints = '0,1,2,3,4,5,6,7,8,9,10,11'  # 临时测试代码，指定多个关节

    # 确定要控制的关节
    if args.joints:
        joint_indices = [int(idx) for idx in args.joints.split(',')]
        test_name = f"speed_change_test_joints"
    else:
        joint_indices = [args.joint_idx]
        test_name = f"speed_change_test_joint"
    
    if args.min_range is not None and args.max_range is not None:
        test_name += f"_range{args.min_range}_{args.max_range}"
    
    # 添加测试模式到名称
    args.test_mode = 2
    test_mode_names = {1: "random", 2: "step", 3: "sweep", 4: "custom"}
    test_name += f"_{test_mode_names[args.test_mode]}"

    # 创建控制器
    controller = Controller(config, test_name, preview=args.preview, visualization=args.visualization)
    
    # 启动性能监控
    controller.start_performance_monitor()

    # 移动到默认位置
    print("移动到默认位置...")
    controller.move_to_default_pos()
    time.sleep(2)  # 等待到达默认位置

    # 配置关节参数
    for joint_idx in joint_indices:
        controller.configure_joint(
            joint_idx, 
            pos_scale=None,  # 速度变化测试不使用位置比例
            speed_scale=controller.config.motion_speeds[joint_idx], 
            min_range=controller.config.motion_lower_bounds[joint_idx] ,
            max_range=controller.config.motion_upper_bounds[joint_idx],
        )
        
        # 设置初始运动模式为下限
        controller.joint_states[joint_idx].anim_state = ANIM_SINE_CUSTOM
        
        print(f"配置关节 {joint_idx}: "
              f"范围=[{args.min_range:.2f}, {args.max_range:.2f}]")
    
    # 生成测试序列
    speed_sequence = generate_test_sequence(args.test_mode, args.steps_per_phase)
    print(f"生成测试序列，总步数: {len(speed_sequence)}")
    
    # 保存测试配置
    test_params = {
        'joint_indices': joint_indices,
        'test_mode': args.test_mode,
        'test_mode_name': test_mode_names[args.test_mode],
        'min_range': args.min_range,
        'max_range': args.max_range,
        'steps_per_phase': args.steps_per_phase,
        'total_steps': len(speed_sequence),
        'speed_sequence': speed_sequence
    }
    controller.save_test_config(test_params)
    
    # 运行控制循环
    running_step = 0
    try:
        print("开始速度变化测试...")
        
        # 第一阶段：速度变化测试
        for speed_scale in speed_sequence:
            running_step += 1
            
            # 运行控制器
            controller.run(
                joint_indices=joint_indices,
                step_count=running_step,
                current_speed_scale=speed_scale
            )
            
            # 定期打印状态
            if running_step % 100 == 0:
                stats = controller.get_performance_stats()
                current_speed = controller.joint_states[joint_indices[0]].speed
                print(f"步数: {running_step}/{len(speed_sequence)}, "
                      f"速度比例: {speed_scale:.2f}, 实际速度: {current_speed:.3f} rad/s, "
                      f"控制频率: {stats.get('control_frequency', 0):.1f} Hz")
        
        print("速度变化测试完成，开始返回默认位置...")
        
        # # 第二阶段：返回默认位置
        # return_steps = 2000
        # for i in range(return_steps):
        #     running_step += 1
            
        #     # 设置低速并返回默认位置
        #     for joint_idx in joint_indices:
        #         controller.joint_states[joint_idx].anim_state = ANIM_SEEK_DEFAULT
        #         controller.joint_states[joint_idx].speed = 0.1 * min(
        #             config.dof_upper[joint_idx] - config.dof_lower[joint_idx],
        #             3.0 * math.pi
        #         )
            
        #     controller.run(
        #         joint_indices=joint_indices,
        #         step_count=running_step
        #     )
            
        #     if i % 100 == 0:
        #         print(f"返回默认位置: {i}/{return_steps}")
       
    except KeyboardInterrupt:
        print("被用户中断")
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        
        if not controller.preview:
            # 保存数据
            print("保存数据...")
            controller.save_all_data()
        
        # 移动到默认位置确保安全
        print("移动到默认位置...")
        controller.move_to_default_pos()
        
        print("所有数据已保存")
        if args.visualization:
            plt.ioff()
            plt.show()
    
    # 打印最终性能统计
    final_stats = controller.get_performance_stats()
    print("\n=== 性能统计 ===")
    print(f"总运行时间: {final_stats.get('total_runtime', 0):.1f} 秒")
    print(f"总步数: {running_step}")
    print(f"平均控制周期: {final_stats.get('avg_control_time', 0)*1000:.1f} 毫秒")
    print(f"控制频率: {final_stats.get('control_frequency', 0):.1f} Hz")
    print(f"测试模式: {test_mode_names[args.test_mode]}")
    
    print("退出") 


if __name__ == "__main__":
    main()
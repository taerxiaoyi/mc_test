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
from config import Config
from base_controller import BaseController
import matplotlib.pyplot as plt
from westlake_sdkpy.core.channel import ChannelFactoryInitialize
ANIM_SEEK_LOWER = 1
ANIM_SEEK_UPPER = 2
ANIM_SEEK_DEFAULT = 3
ANIM_FINISHED = 4
ANIM_SEEK_SCALE = 5

class Controller(BaseController):
    def __init__(self, config: Config, test_name: str, preview=True, visualization=False) -> None:
        super().__init__(config, test_name, preview, visualization)

    def run(self, joint_indices: Union[int, List[int]], step_count: int, 
            modes: Optional[Dict[int, int]] = None, current_speed_scale: float = None) -> bool:
        """
        运行控制循环
        
        Args:
            joint_indices: 单个关节索引或关节索引列表
            step_count: 步数计数
            modes: 每个关节的运动模式字典 {joint_idx: anim_state}
            current_speed_scale: 当前速度比例
            
        Returns:
            是否继续运行
        """
        if isinstance(joint_indices, int):
            joint_indices = [joint_indices]
        
        if modes is None:
            # 使用每个关节的当前状态
            modes = {joint_idx: self.joint_states[joint_idx].anim_state for joint_idx in joint_indices}
        
        # 更新速度比例
        if current_speed_scale is not None:
            for joint_idx in joint_indices:
                range_size = self.config.dof_upper[joint_idx] - self.config.dof_lower[joint_idx]
                self.joint_states[joint_idx].speed = current_speed_scale * min(range_size, 3.0 * math.pi)
        
        if self.preview:
            # 仿真模式
            sim_state = self.e1_sim.read_state()
            target_dof_pos, dof_deltas = self.cmd_joints(
                sim_state['dof_pos'].copy(), 
                target_joints=joint_indices,
                modes=modes
            )
            
            # 计算所有目标关节的误差和扭矩
            for joint_idx in joint_indices:
                position_error = target_dof_pos[joint_idx] - sim_state['dof_pos'][joint_idx]
                velocity_error = (dof_deltas[joint_idx] / self.config.control_dt) - sim_state['dof_vel'][joint_idx]
                torque = self.compute_torque(position_error, velocity_error, joint_idx)
                
                # 记录数据
                self.add_joint_data_point(
                    joint_idx,
                    target_dof_pos[joint_idx],
                    sim_state['dof_pos'][joint_idx],
                    dof_deltas[joint_idx] / self.config.control_dt,
                    sim_state['dof_vel'][joint_idx],
                    torque,
                    sim_state['dof_torque'][joint_idx] if 'dof_torque' in sim_state else 0.0,
                    self.config.kps[joint_idx],
                    self.config.kds[joint_idx],
                    self.joint_states[joint_idx].speed
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
            
            # 发送控制命令到真实机器人
            self.e1_real.step(target_dof_pos.copy(), kps=self.config.kps, kds=self.config.kds)
            
            # 读取实际状态
            real_state = self.e1_real.read_state()
            
            # 计算所有目标关节的误差和扭矩
            target_positions = []
            actual_positions = []
            target_velocities = []
            actual_velocities = []
            compute_torques = []
            actual_torques = []
            kps = []
            kds = []
            speeds = []
            
            for joint_idx in joint_indices:
                position_error = target_dof_pos[joint_idx] - real_state['dof_pos'][joint_idx]
                velocity_error = (dof_deltas[joint_idx] / self.config.control_dt) - real_state['dof_vel'][joint_idx]
                computed_torque = self.compute_torque(position_error, velocity_error, joint_idx)
                actual_torque = real_state['dof_torque'][joint_idx]
                
                target_positions.append(target_dof_pos[joint_idx])
                actual_positions.append(real_state['dof_pos'][joint_idx])
                target_velocities.append(dof_deltas[joint_idx] / self.config.control_dt)
                actual_velocities.append(real_state['dof_vel'][joint_idx])
                compute_torques.append(computed_torque)
                actual_torques.append(actual_torque)
                kps.append(self.config.kps[joint_idx])
                kds.append(self.config.kds[joint_idx])
                speeds.append(self.joint_states[joint_idx].speed)
            
            # 记录多关节数据
            self.add_multi_joint_data(
                joint_indices,
                target_positions,
                actual_positions,
                target_velocities,
                actual_velocities,
                compute_torques,
                actual_torques,
                kps,
                kds,
                speeds
            )
            
            # 同时更新仿真（用于可视化）
            self.e1_sim.step(real_state.copy())
        
        # 限速
        self.rate_limiter.sleep()
        
        # 记录控制时间（用于性能监控）
        self.log_control_time()
        
        # 更新可视化
        if self.visualization:
            self.update_visualization(step_count)
        
        return True


def generate_velocity_test_sequence(test_type: str, steps_per_speed: int = 2000) -> List[float]:
    """
    生成速度测试序列
    
    Args:
        test_type: 测试类型 - "step" | "random" | "sweep"
        steps_per_speed: 每个速度测试的步数
        
    Returns:
        速度比例序列
    """
    if test_type == "step":
        # 阶跃测试：低→中→高→低
        speed_sequence = []
        for speed in [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
            speed_sequence.extend([speed] * steps_per_speed)
    elif test_type == "random":
        # 随机测试：每100步随机变化速度
        speed_sequence = []
        for i in range(7 * steps_per_speed // 100):
            speed_sequence.extend([np.random.uniform(0.01, 2.0)] * 100)
    elif test_type == "sweep":
        # 扫频测试：速度从低到高再到低
        speed_sequence = []
        # 上升阶段
        for i in range(20):
            speed = 0.1 + 0.095 * i  # 0.1, 0.195, 0.29, ..., 2.0
            speed_sequence.extend([speed] * (steps_per_speed // 20))
        # 下降阶段
        for i in range(20):
            speed = 2.0 - 0.095 * i  # 2.0, 1.905, 1.81, ..., 0.1
            speed_sequence.extend([speed] * (steps_per_speed // 20))
    else:
        print(f"未知测试类型: {test_type}，使用默认阶跃测试")
        speed_sequence = []
        for speed in [0.1, 0.5, 1.0, 2.0]:
            speed_sequence.extend([speed] * steps_per_speed)
    
    return speed_sequence


def main():
    import argparse

    parser = argparse.ArgumentParser(description="机器人速度跟踪测试")
    parser.add_argument("--config", type=str, help="configs文件夹中的配置文件名称", default="o1.yaml")
    parser.add_argument("--joint_idx", type=int, help="要控制的关节索引", default=0)
    parser.add_argument("--joints", type=str, help="要控制的多个关节索引，用逗号分隔，例如：0,1,2", default=None)
    parser.add_argument("--kp", type=float, help="比例增益", default=None)
    parser.add_argument("--kd", type=float, help="微分增益", default=None)
    parser.add_argument("--min_range", type=float, help="最小关节位置(弧度)", default=-2)
    parser.add_argument("--max_range", type=float, help="最大关节位置(弧度)", default=2)
    parser.add_argument("--test_type", type=str, choices=["step", "random", "sweep"], 
                       help="速度测试类型: step-阶跃, random-随机, sweep-扫频", default="step")
    parser.add_argument("--steps_per_speed", type=int, help="每个速度测试的步数", default=2000)
    parser.add_argument("--preview", type=bool, default=True, help="在预览模式（仿真）下运行")
    parser.add_argument("--visualization", action="store_true", help="启用可视化")
    
    args = parser.parse_args()

    ChannelFactoryInitialize(0)

    # 加载配置
    config_path = f"robot_test/{args.config}"
    config = Config(config_path)
    
    # 确定要控制的关节
    if args.joints:
        joint_indices = [int(idx) for idx in args.joints.split(',')]
        test_name = f"velocity_test_joints_{args.joints.replace(',', '_')}"
    else:
        joint_indices = [args.joint_idx]
        test_name = f"velocity_test_joint_{args.joint_idx}"
    
    # 添加测试类型到名称
    test_name += f"_{args.test_type}"
    
    # 添加参数到测试名称
    if args.kp is not None:
        test_name += f"_kp{args.kp}"
    if args.kd is not None:
        test_name += f"_kd{args.kd}"

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
            pos_scale=None,  # 速度测试不使用位置比例
            speed_scale=0.1,  # 初始速度比例
            kp=args.kp, 
            kd=args.kd, 
            min_range=args.min_range,
            max_range=args.max_range
        )
        
        # 设置初始运动模式为下限
        controller.joint_states[joint_idx].anim_state = ANIM_SEEK_LOWER
        
        print(f"配置关节 {joint_idx}: "
              f"KP={args.kp if args.kp else config.kps[joint_idx]:.2f}, "
              f"KD={args.kd if args.kd else config.kds[joint_idx]:.2f}, "
              f"范围=[{args.min_range:.2f}, {args.max_range:.2f}]")
    
    # 生成速度测试序列
    speed_sequence = generate_velocity_test_sequence(args.test_type, args.steps_per_speed)
    
    # 保存测试配置
    test_params = {
        'joint_indices': joint_indices,
        'test_type': args.test_type,
        'kp': args.kp,
        'kd': args.kd,
        'min_range': args.min_range,
        'max_range': args.max_range,
        'steps_per_speed': args.steps_per_speed,
        'total_steps': len(speed_sequence),
        'speed_sequence': speed_sequence[:100] if len(speed_sequence) > 100 else speed_sequence  # 只保存前100个速度值
    }
    controller.save_test_config(test_params)
    
    print(f"开始{args.test_type}速度测试，共{len(speed_sequence)}步")
    
    # 运行测试
    running_step = 0
    try:
        for step, speed_scale in enumerate(speed_sequence):
            running_step = step + 1
            
            # 执行控制步骤
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
        
        print("速度测试完成，开始返回默认位置...")
        
        # 返回默认位置
        return_steps = 500
        for i in range(return_steps):
            running_step += 1
            
            # 设置低速并返回默认位置
            for joint_idx in joint_indices:
                controller.joint_states[joint_idx].anim_state = ANIM_SEEK_DEFAULT
                range_size = config.dof_upper[joint_idx] - config.dof_lower[joint_idx]
                controller.joint_states[joint_idx].speed = 0.1 * min(range_size, 3.0 * math.pi)
            
            controller.run(
                joint_indices=joint_indices,
                step_count=running_step
            )
            
            if i % 100 == 0:
                print(f"返回默认位置: {i}/{return_steps}")
                
    except KeyboardInterrupt:
        print("被用户中断")
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
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
    print(f"测试类型: {args.test_type}")
    
    print("速度跟踪测试完成")


if __name__ == "__main__":
    main()
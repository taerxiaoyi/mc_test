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

class Controller(BaseController):
    def __init__(self, config: Config, test_name: str, preview=True, visualization=False) -> None:
        super().__init__(config, test_name, preview, visualization)
        # target_position 已经在BaseController的JointState中管理了
        # self.target_position = None

    def run(self, joint_indices: Union[int, List[int]], step_count: int, 
            modes: Optional[Dict[int, int]] = None):
        """
        运行控制循环
        
        Args:
            joint_indices: 单个关节索引或关节索引列表
            step_count: 步数计数
            modes: 每个关节的运动模式字典 {joint_idx: anim_state}
        """
        if isinstance(joint_indices, int):
            joint_indices = [joint_indices]
        
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


def main():
    import argparse

    parser = argparse.ArgumentParser(description="机器人关节控制测试")
    parser.add_argument("--config", type=str, help="configs文件夹中的配置文件名称", default="o1.yaml")
    parser.add_argument("--pos_scale", type=float, help="0.0到1.0之间的比例值", default=0.7)
    parser.add_argument("--joint_idx", type=int, help="要控制的关节索引", default=0)
    parser.add_argument("--joints", type=str, help="要控制的多个关节索引，用逗号分隔，例如：0,1,2", default=None)
    parser.add_argument("--speed_scale", type=float, help="速度比例因子", default=None)
    parser.add_argument("--preview",type=bool,default=True, help="在预览模式（仿真）下运行")
    parser.add_argument("--visualization", action="store_true", help="启用可视化")
    parser.add_argument("--mode", type=str, choices=["lower", "upper", "default", "scale"], 
                       help="运动模式", default="scale")
    parser.add_argument("--duration", type=int, help="测试持续时间（步数）", default=4000)
    args = parser.parse_args()

    ChannelFactoryInitialize(0)

    # 加载配置
    config_path = f"test/{args.config}"
    config = Config(config_path)

    args.joints = '0,1,2,4,5'  # 临时测试代码，指定多个关节
    # args.joints = '0,1,2,3,4,5,6,7,8,9,10,11'  # 临时测试代码，指定多个关节
    # args.joints = '0,2,3,4,5,7,8,9,10'  # 临时测试代码，指定多个关节
    # args.joints = '11,12,13'  # 临时测试代码，指定多个关节
    # args.joints = '0,1,2,3,4,5,6,7,8,9,10,11,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28'  # 临时测试代码，取消多个关节指定
    
    # 确定要控制的关节
    if args.joints:
        joint_indices = [int(idx) for idx in args.joints.split(',')]
        test_name = f"hold_test_joints"
    else:
        joint_indices = [args.joint_idx]
        test_name = f"hold_test_joint"
    
    # 创建控制器
    controller = Controller(config, test_name, preview=args.preview, visualization=args.visualization)
    
    # 启动性能监控
    controller.start_performance_monitor()

    # 移动到默认位置
    print("移动到默认位置...")
    controller.move_to_default_pos()
    time.sleep(2)  # 等待到达默认位置

    # 配置关节参数
    mode_mapping = {
        "lower": ANIM_SEEK_LOWER,
        "upper": ANIM_SEEK_UPPER, 
        "default": ANIM_SEEK_DEFAULT,
        "scale": ANIM_SEEK_SCALE
    }
    
    target_modes = {}
    for joint_idx in joint_indices:
        # 配置关节参数
        controller.configure_joint(
            joint_idx, 
            pos_scale= controller.config.hold_pos[joint_idx],
            speed_scale=controller.config.motion_speeds[joint_idx],
        )
        
        # 设置运动模式
        target_modes[joint_idx] = mode_mapping[args.mode]
        
        print(f"配置关节 {joint_idx}: 模式={args.mode}, "
              f"位置比例={args.pos_scale if args.mode == 'scale' else 'N/A'}, ")
             
    # 保存测试配置
    test_params = {
        'joint_indices': joint_indices,
        'mode': args.mode,
        'pos_scale': args.pos_scale,
        'speed_scale': args.speed_scale,
        'duration': args.duration
    }
    controller.save_test_config(test_params)
    
    # 运行控制循环
    running_step = 0
    try:
        print("开始控制循环...")
        while running_step < args.duration:
            running_step += 1
            
            # 运行控制器
            controller.run(
                joint_indices=joint_indices,
                step_count=running_step,
                modes=target_modes
            )
            
            # 定期打印状态
            if running_step % 100 == 0:
                stats = controller.get_performance_stats()
                print(f"步数: {running_step}/{args.duration}, "
                      f"控制频率: {stats.get('control_frequency', 0):.1f} Hz")
                
                # 打印每个关节的状态
                for joint_idx in joint_indices:
                    joint_state = controller.joint_states[joint_idx]
                    current_pos = controller.e1_sim.read_state()['dof_pos'][joint_idx]
                    print(f"  关节 {joint_idx}: 位置={current_pos:.3f} rad, "
                          f"状态={joint_state.anim_state}, 速度={joint_state.speed:.3f} rad/s")
        print("移动到默认位置...")
        controller.move_to_default_pos()    
      
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
    print(f"平均控制周期: {final_stats.get('avg_control_time', 0)*1000:.1f} 毫秒")
    print(f"控制频率: {final_stats.get('control_frequency', 0):.1f} Hz")
    print(f"最大控制周期: {final_stats.get('max_control_time', 0)*1000:.1f} 毫秒")
    print(f"最小控制周期: {final_stats.get('min_control_time', 0)*1000:.1f} 毫秒")
    
    print("退出") 


if __name__ == "__main__":
    main()
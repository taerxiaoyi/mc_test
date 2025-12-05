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
# from westlake_sdkpy.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize

class Controller(BaseController):
    def __init__(self, config: Config, test_name: str, visualization: bool = False) -> None:
        super().__init__(config, test_name, visualization)
        self.state_read_count = 0
        self.position_history = {i: [] for i in range(config.num_joints)}
        self.velocity_history = {i: [] for i in range(config.num_joints)}
        self.torque_history = {i: [] for i in range(config.num_joints)}

    def run(self, step_count: int, test_mode: str = "state_read") -> bool:
        """
        运行状态读取测试
        
        Args:
            step_count: 步数计数
            test_mode: 测试模式 - "state_read" | "limit_check" | "damping_test"
            
        Returns:
            是否继续运行
        """
        self.state_read_count += 1
        
   
        # 真实机器人模式
        real_state = self.e1_real.read_state()
            
        # 记录真实状态数据
        target_positions = []
        actual_positions = []
        target_velocities = []
        actual_velocities = []
        compute_torques = []
        actual_torques = []
        kps = []
        kds = []
        speeds = []
            
        for joint_idx in range(self.config.num_joints):
            # 保存历史数据
            self.position_history[joint_idx].append(real_state['dof_pos'][joint_idx])
            self.velocity_history[joint_idx].append(real_state['dof_vel'][joint_idx])
            if 'dof_torque' in real_state:
                self.torque_history[joint_idx].append(real_state['dof_torque'][joint_idx])
            
            target_positions.append(real_state['dof_pos'][joint_idx])  # 目标位置设为实际位置
            actual_positions.append(real_state['dof_pos'][joint_idx])
            target_velocities.append(0.0)  # 目标速度设为0
            actual_velocities.append(real_state['dof_vel'][joint_idx])
            compute_torques.append(0.0)  # 计算扭矩设为0
            actual_torques.append(real_state['dof_torque'][joint_idx] if 'dof_torque' in real_state else 0.0)
            kps.append(self.config.kps[joint_idx])
            kds.append(self.config.kds[joint_idx])
            speeds.append(self.joint_states[joint_idx].speed)
            
        # 记录多关节数据
        self.add_multi_joint_data(
            list(range(self.config.num_joints)),
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
            
        # 定期打印状态
        if step_count % 100 == 0:
            self.print_joint_states(real_state)
        
        # 同步到仿真用于可视化
        self.e1_sim.step(real_state.copy())
        self.rate_limiter.sleep()
        
        # 记录控制时间（用于性能监控）
        self.log_control_time()
        
        # 更新可视化
        if self.visualization:
            self.update_visualization(step_count)
        
        return True
    
    def print_joint_states(self, state: Dict):
        """打印关节状态"""
        print(f"\n=== 状态读取计数: {self.state_read_count} ===")
        print("真实机器人状态:")
        
        # 按照关节分组打印
        groups = [
            (0, 6, "右腿"),
            (6, 12, "左腿"), 
            (12, 15, "腰部"),
            (15, 22, "右臂"),
            (22, 29, "左臂")
        ]
        
        for start, end, name in groups:
            print(f"\n{name}关节:")
            for i in range(start, end):
                pos = state['dof_pos'][i]
                vel = state['dof_vel'][i]
                torque = state['dof_torque'][i] if 'dof_torque' in state else 0.0
                print(f"  关节 {i}: 位置={pos:6.3f} rad, 速度={vel:6.3f} rad/s, 扭矩={torque:6.2f} Nm")
    
    def check_joint_limits(self, state: Dict) -> Dict:
        """
        检查关节限位
        
        Args:
            state: 机器人状态
            
        Returns:
            限位检查结果字典
        """
        limit_results = {}
        
        for joint_idx in range(self.config.num_joints):
            current_pos = state['dof_pos'][joint_idx]
            lower_limit = self.config.dof_lower[joint_idx]
            upper_limit = self.config.dof_upper[joint_idx]
            
            # 检查是否接近或超出限位
            if current_pos <= lower_limit + 0.1:  # 接近下限
                limit_results[joint_idx] = {
                    'status': 'NEAR_LOWER_LIMIT',
                    'position': current_pos,
                    'limit': lower_limit,
                    'distance': current_pos - lower_limit
                }
            elif current_pos >= upper_limit - 0.1:  # 接近上限
                limit_results[joint_idx] = {
                    'status': 'NEAR_UPPER_LIMIT', 
                    'position': current_pos,
                    'limit': upper_limit,
                    'distance': upper_limit - current_pos
                }
            elif current_pos < lower_limit:  # 超出下限
                limit_results[joint_idx] = {
                    'status': 'EXCEED_LOWER_LIMIT',
                    'position': current_pos,
                    'limit': lower_limit,
                    'distance': current_pos - lower_limit
                }
            elif current_pos > upper_limit:  # 超出上限
                limit_results[joint_idx] = {
                    'status': 'EXCEED_UPPER_LIMIT',
                    'position': current_pos,
                    'limit': upper_limit, 
                    'distance': upper_limit - current_pos
                }
            else:
                limit_results[joint_idx] = {
                    'status': 'WITHIN_LIMITS',
                    'position': current_pos,
                    'limit_lower': lower_limit,
                    'limit_upper': upper_limit
                }
        
        return limit_results
    
    def analyze_state_data(self) -> Dict:
        """分析状态数据"""
        analysis = {}
        
        for joint_idx in range(self.config.num_joints):
            if self.position_history[joint_idx]:
                positions = np.array(self.position_history[joint_idx])
                velocities = np.array(self.velocity_history[joint_idx])
                torques = np.array(self.torque_history[joint_idx]) if self.torque_history[joint_idx] else np.array([0.0])
                
                analysis[joint_idx] = {
                    'position_stats': {
                        'mean': np.mean(positions),
                        'std': np.std(positions),
                        'min': np.min(positions),
                        'max': np.max(positions),
                        'range': np.ptp(positions)
                    },
                    'velocity_stats': {
                        'mean': np.mean(velocities),
                        'std': np.std(velocities),
                        'min': np.min(velocities),
                        'max': np.max(velocities)
                    },
                    'torque_stats': {
                        'mean': np.mean(torques),
                        'std': np.std(torques),
                        'min': np.min(torques),
                        'max': np.max(torques)
                    } if len(torques) > 1 else {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
                }
        
        return analysis


def main():
    import argparse

    parser = argparse.ArgumentParser(description="机器人状态读取测试")
    parser.add_argument("--config", type=str, help="configs文件夹中的配置文件名称", default="g1.yaml")
    parser.add_argument("--test_mode", type=str, choices=['state_read', 'limit_check', 'damping_test'], 
                       default='state_read', help="测试模式")
    parser.add_argument("--duration", type=int, help="测试持续时间（秒）", default=3000)
    parser.add_argument("--visualization", action="store_true", help="启用可视化")
    
    args = parser.parse_args()

    ChannelFactoryInitialize(0)
    
    # 加载配置
    config_path = f"test/{args.config}"
    config = Config(config_path)
    
    # 创建控制器
    controller = Controller(config, 'state_read_test', visualization=args.visualization)
    
    # 启动性能监控
    controller.start_performance_monitor()

    print("🚀 开始状态读取测试")
    print(f"测试模式: {args.test_mode}")
    print(f"持续时间: {args.duration} 秒")
    
    # 根据测试模式执行不同的初始化
    if args.test_mode == "state_read" or args.test_mode == "limit_check":
        print("📊 测试1: 真机状态返回值验证")
        print("   检测方法: 真机进入0力矩状态，随意摆动机器人，记录状态数据")
        # controller.zero_torque_state()
        
    elif args.test_mode == "damping_test":
        print("🔄 测试3: 关节阻尼模式测试")
        print("   检测方法: 进入阻尼模式，随意摆动机器人，观察阻尼效果")
        controller.damping_mode()
    
    # 保存测试配置
    test_params = {
        'test_mode': args.test_mode,
        'duration': args.duration,
        'visualization': args.visualization
    }
    controller.save_test_config(test_params)
    
    # 运行测试
    running_step = 0
    start_time = time.time()
    
    try:
        while time.time() - start_time < args.duration:
            running_step += 1
            
            # 运行控制器
            controller.run(step_count=running_step, test_mode=args.test_mode)
            
            # 定期打印性能信息
            if running_step % 500 == 0:
                stats = controller.get_performance_stats()
                elapsed = time.time() - start_time
                print(f"运行时间: {elapsed:.1f}s/{args.duration}s, "
                      f"控制频率: {stats.get('control_frequency', 0):.1f} Hz, "
                      f"状态读取次数: {controller.state_read_count}")
                
    except KeyboardInterrupt:
        print("测试被用户中断")
    except Exception as e:
        print(f"测试出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 分析状态数据
        print("\n正在分析状态数据...")
        analysis = controller.analyze_state_data()
        
        # 打印分析结果
        print(f"\n{'='*60}")
        print("状态数据分析报告")
        print(f"{'='*60}")
        
        for joint_idx, data in analysis.items():
            pos_stats = data['position_stats']
            vel_stats = data['velocity_stats']
            torque_stats = data['torque_stats']
            
            print(f"\n关节 {joint_idx}:")
            print(f"  位置 - 均值: {pos_stats['mean']:.3f}rad, 标准差: {pos_stats['std']:.3f}, "
                  f"范围: [{pos_stats['min']:.3f}, {pos_stats['max']:.3f}]")
            print(f"  速度 - 均值: {vel_stats['mean']:.3f}rad/s, 标准差: {vel_stats['std']:.3f}")
            print(f"  扭矩 - 均值: {torque_stats['mean']:.2f}Nm, 最大值: {torque_stats['max']:.2f}Nm")
        
        # 限位检查（如果是限位测试模式）
        if args.test_mode == "limit_check":
            print(f"\n{'='*60}")
            print("关节限位检查报告")
            print(f"{'='*60}")
            
            real_state = controller.e1_real.read_state()
            limit_results = controller.check_joint_limits(real_state)
            
            for joint_idx, result in limit_results.items():
                status = result['status']
                if status == 'WITHIN_LIMITS':
                    print(f"关节 {joint_idx}: ✅ 在限位范围内")
                elif status == 'NEAR_LOWER_LIMIT':
                    print(f"关节 {joint_idx}: ⚠️  接近下限, 距离: {result['distance']:.3f}rad")
                elif status == 'NEAR_UPPER_LIMIT':
                    print(f"关节 {joint_idx}: ⚠️  接近上限, 距离: {result['distance']:.3f}rad")
                elif status == 'EXCEED_LOWER_LIMIT':
                    print(f"关节 {joint_idx}: ❗ 超出下限, 超出: {abs(result['distance']):.3f}rad")
                elif status == 'EXCEED_UPPER_LIMIT':
                    print(f"关节 {joint_idx}: ❗ 超出上限, 超出: {abs(result['distance']):.3f}rad")
        
        # 保存数据
        print("\n保存数据...")
        controller.save_all_data()
        
        # 打印最终性能统计
        final_stats = controller.get_performance_stats()
        print(f"\n=== 性能统计 ===")
        print(f"总运行时间: {final_stats.get('total_runtime', 0):.1f} 秒")
        print(f"总步数: {running_step}")
        print(f"状态读取次数: {controller.state_read_count}")
        print(f"平均控制周期: {final_stats.get('avg_control_time', 0)*1000:.1f} 毫秒")
        print(f"控制频率: {final_stats.get('control_frequency', 0):.1f} Hz")
        
        if args.visualization:
            plt.ioff()
            plt.show()
    
    print("状态读取测试完成")


if __name__ == "__main__":
    main()
import sys
import os
import time
import threading
import csv
import math
import json
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from collections import deque
from typing import Union, List, Dict, Tuple, Optional, Any

from loop_rate_limiters import RateLimiter
from sim_robot import E1SimEnv
from real_robot import E1RealEnv
from config import Config
from westlake_sdkpy.core.channel import ChannelFactoryInitialize
from utils import DataSaverThread
from utils import VisualizationThread
# Animation state constants
ANIM_SEEK_LOWER = 1
ANIM_SEEK_UPPER = 2
ANIM_SEEK_DEFAULT = 3
ANIM_FINISHED = 4
ANIM_SEEK_SCALE = 5
ANIM_SINE_CUSTOM = 6
ANIM_S_CURVE = 7   # S曲线运动
ANIM_TRAPEZOIDAL_VEL = 8   # 梯形速度运动 
ANIM_TRIANGLE_WAVE = 9      # 三角波运动
ANIM_SQUARE_WAVE = 10 

class JointState:
    """单个关节的状态管理类"""
    
    def __init__(self, joint_idx: int, config: Config):
        """
        初始化关节状态
        
        Args:
            joint_idx: 关节索引
            config: 配置对象
        """
        self.joint_idx = joint_idx
        self.anim_state = ANIM_SEEK_LOWER
        self.target_position = None
        
        # 计算默认速度
        range_size = config.dof_upper[joint_idx] - config.dof_lower[joint_idx]
        self.speed = 0.1 * min(range_size, 3.0 * math.pi)
        
        # 原始设置记录
        self.original_settings = {}
        
        # 性能数据
        self.control_times = deque(maxlen=100)
        self.last_control_time = time.time()


class BaseController:
    """改进的基础控制器类，提供机器人控制功能"""
    
    def __init__(self, config: Config, test_name: str, preview=True, visualization=False):
        """
        初始化基础控制器
        
        Args:
            config: 配置对象
            test_name: 测试名称
            preview: 是否在预览模式（仿真）下运行
            visualization: 是否启用数据曲线可视化
        """
        self.config = config
        self.preview = preview
        self.visualization = visualization
        
        # 初始化仿真和真实环境
        fps = int(1.0 / self.config.control_dt)
        self.e1_sim = E1SimEnv(config=config, fps=fps, kps=self.config.kps, kds=self.config.kds)
        self.e1_real = E1RealEnv(config=config)
        self.rate_limiter = RateLimiter(frequency=fps, warn=True)
        
        # 初始化关节状态
        self.joint_states = [JointState(i, config) for i in range(config.num_joints)]
        self.global_anim_state = ANIM_SEEK_LOWER
        
        # 性能监控
        self.performance_data = {
            'control_times': deque(maxlen=100),
            'last_control_time': time.time()
        }

        # 初始化数据记录结构
        self.last_control_data = None
        self.joint_data = {}
        for i in range(config.num_joints):
            self.joint_data[i] = {
                'time': [],
                'target_pos': [],
                'actual_pos': [],
                'target_vel': [],
                'actual_vel': [],
                'compute_torque': [],
                'actual_torque': [],
                'kp': [],
                'kd': [],
                'speed': [],
                'error': []
            }
            
        self.data_lock = threading.Lock()
        self.start_time = time.time()
        
        # 设置数据保存目录和文件
        self.data_dir = "tracking_data/" + time.strftime('%Y%m%d')+"/" + time.strftime('%H%M%S')
        os.makedirs(self.data_dir, exist_ok=True)
        self.data_file_name = os.path.join(
            self.data_dir, 
            f"{test_name}"
        )
        self.data_file = self.data_file_name + '.csv'
        
        # 初始化数据保存线程
        self.data_saver = DataSaverThread(self.data_file, config)
        self.data_saver.start()
        
        print(f"数据将保存到: {self.data_file}")
        
        # 测试配置
        self.test_config = {
            'test_name': test_name,
            'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'config_file': config.config_path if hasattr(config, 'config_path') else 'unknown',
            'preview_mode': preview,
            'visualization': visualization
        }
        
        # 初始化可视化线程
        if self.visualization:
            self.visualization_thread = VisualizationThread(config, self.joint_data, self.data_lock)
            self.visualization_thread.start()
            print(f"可视化将每 {self.visualization_thread.visualization_interval} 步更新一次")
        else:
            self.visualization_thread = None
    
    
    
    def add_joint_data_point(self, joint_idx: int, target_pos: float, actual_pos: float, 
                           target_vel: float, actual_vel: float, compute_torque: float, 
                           actual_torque: float, kp: float, kd: float, speed: float):
        """为单个关节添加数据点"""
        current_time = time.time() - self.start_time
        error = actual_pos - target_pos
        
        # 线程安全的数据添加
        with self.data_lock:
            if joint_idx in self.joint_data:
                data = self.joint_data[joint_idx]
                data['time'].append(current_time)
                data['target_pos'].append(target_pos)
                data['actual_pos'].append(actual_pos)
                data['target_vel'].append(target_vel)
                data['actual_vel'].append(actual_vel)
                data['compute_torque'].append(compute_torque)
                data['actual_torque'].append(actual_torque)
                data['kp'].append(kp)
                data['kd'].append(kd)
                data['speed'].append(speed)
                data['error'].append(error)
        
        # 通过线程保存到CSV文件
        data_row = [current_time, joint_idx, target_pos, actual_pos, target_vel, actual_vel, 
                   compute_torque, actual_torque, kp, kd, speed, error]
        self.data_saver.add_data(data_row)
    
    def add_multi_joint_data(self, joint_indices: List[int], target_positions: List[float], 
                           actual_positions: List[float], target_velocities: List[float], 
                           actual_velocities: List[float], compute_torques: List[float], 
                           actual_torques: List[float], kps: List[float], kds: List[float], 
                           speeds: List[float]):
        """为多个关节添加数据点"""
        current_time = time.time() - self.start_time
        
        for i, joint_idx in enumerate(joint_indices):
            error = actual_positions[i] - target_positions[i]
            
            # 线程安全的数据添加
            with self.data_lock:
                if joint_idx in self.joint_data:
                    data = self.joint_data[joint_idx]
                    data['time'].append(current_time)
                    data['target_pos'].append(target_positions[i])
                    data['actual_pos'].append(actual_positions[i])
                    data['target_vel'].append(target_velocities[i])
                    data['actual_vel'].append(actual_velocities[i])
                    data['compute_torque'].append(compute_torques[i])
                    data['actual_torque'].append(actual_torques[i])
                    data['kp'].append(kps[i])
                    data['kd'].append(kds[i])
                    data['speed'].append(speeds[i])
                    data['error'].append(error)
            
            # 通过线程保存到CSV文件
            data_row = [
                current_time, joint_idx, target_positions[i], actual_positions[i],
                target_velocities[i], actual_velocities[i], compute_torques[i],
                actual_torques[i], kps[i], kds[i], speeds[i], error
            ]
            self.data_saver.add_data(data_row)
    
    def update_visualization(self, step_count: int):
        """请求更新可视化（非阻塞）"""
        if self.visualization_thread:
            performance_stats = self.get_performance_stats()
            self.visualization_thread.request_update(step_count, performance_stats)
    
    def save_test_config(self, test_params: Dict[str, Any] = None):
        """保存测试配置"""
        if test_params is None:
            test_params = {}
        
        # 合并测试参数
        config_to_save = {**self.test_config, **test_params}
        
        # 添加性能统计
        config_to_save['performance_stats'] = self.get_performance_stats()
        config_to_save['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        
        config_file = self.data_file_name + '_config.json'
        with open(config_file, 'w') as f:
            json.dump(config_to_save, f, indent=2, default=str)
        
        print(f"测试配置保存为: {config_file}")
    
    def load_test_config(self, config_path: str) -> Dict[str, Any]:
        """加载测试配置"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def save_all_data(self):
        """保存所有数据到文件"""
        # 停止数据保存线程
        if hasattr(self, 'data_saver'):
            self.data_saver.stop()
            self.data_saver.join()
        
        # 停止可视化线程
        if self.visualization_thread:
            self.visualization_thread.stop()
            self.visualization_thread.join()
        
        # 线程安全的数据检索
        with self.data_lock:
            joint_data_copy = {}
            for joint_idx, data in self.joint_data.items():
                joint_data_copy[joint_idx] = {
                    key: value.copy() for key, value in data.items()
                }
        
        # 创建汇总图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        
        # 位置跟踪图表
        for joint_idx, data in joint_data_copy.items():
            if data['time'] and joint_idx < len(colors):
                ax1.plot(data['time'], data['target_pos'], 
                        f'{colors[joint_idx]}-', label=f'Joint {joint_idx} Target', 
                        linewidth=1, alpha=0.7)
                ax1.plot(data['time'], data['actual_pos'], 
                        f'{colors[joint_idx]}--', label=f'Joint {joint_idx} Actual', 
                        linewidth=1, alpha=0.7)
        
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Position (rad)')
        ax1.set_title('Position Tracking - Complete Data')
        ax1.legend()
        ax1.grid(True)
        
        # 速度跟踪图表
        for joint_idx, data in joint_data_copy.items():
            if data['time'] and joint_idx < len(colors):
                ax2.plot(data['time'], data['target_vel'], 
                        f'{colors[joint_idx]}-', label=f'Joint {joint_idx} Target', 
                        linewidth=1, alpha=0.7)
                ax2.plot(data['time'], data['actual_vel'], 
                        f'{colors[joint_idx]}--', label=f'Joint {joint_idx} Actual', 
                        linewidth=1, alpha=0.7)
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Velocity (rad/s)')
        ax2.set_title('Velocity Tracking - Complete Data')
        ax2.legend()
        ax2.grid(True)
        
        # 扭矩图表
        for joint_idx, data in joint_data_copy.items():
            if data['time'] and joint_idx < len(colors):
                ax3.plot(data['time'], data['compute_torque'], 
                        f'{colors[joint_idx]}-', label=f'Joint {joint_idx} Computed', 
                        linewidth=1, alpha=0.7)
                ax3.plot(data['time'], data['actual_torque'], 
                        f'{colors[joint_idx]}--', label=f'Joint {joint_idx} Actual', 
                        linewidth=1, alpha=0.7)
        
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Torque (Nm)')
        ax3.set_title('Torque - Complete Data')
        ax3.legend()
        ax3.grid(True)
        
        # 误差图表
        for joint_idx, data in joint_data_copy.items():
            if data['time'] and joint_idx < len(colors):
                ax4.plot(data['time'], data['error'], 
                        f'{colors[joint_idx]}-', label=f'Joint {joint_idx} Error', 
                        linewidth=1, alpha=0.7)
        
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Error (rad)')
        ax4.set_title('Position Error - Complete Data')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        
        # 保存图表
        summary_plot = self.data_file_name + '.png'
        plt.savefig(summary_plot, dpi=150)
        print(f"汇总图表保存为: {summary_plot}")
        plt.close(fig)
        
        # 保存数据为NPZ文件
        npz_file = self.data_file_name + '.npz'
        np.savez(npz_file, **{f'joint_{idx}': data for idx, data in joint_data_copy.items()})
        print(f"数据保存为: {npz_file}")
        
        # 保存测试配置
        self.save_test_config()
    
    def configure_joint(self, joint_idx: int, 
                   pos_scale: Optional[float] = None,
                   speed_scale: Optional[float] = None,
                   kp: Optional[float] = None,
                   kd: Optional[float] = None,
                   min_range: Optional[float] = None,
                   max_range: Optional[float] = None) -> Dict[str, Any]:
        """
        配置关节参数
        
        Args:
            joint_idx: 要配置的关节索引
            pos_scale: 位置比例 (0.0 到 1.0)
            speed_scale: 速度比例因子
            kp: 比例增益
            kd: 微分增益
            min_range: 最小范围限制
            max_range: 最大范围限制
            
        Returns:
            每个被更改参数的原始值字典
        """
        if joint_idx < 0 or joint_idx >= self.config.num_joints:
            raise ValueError(f"无效的关节索引: {joint_idx}")
        
        joint_state = self.joint_states[joint_idx]
        original_values = {}
        
        if pos_scale is not None:
            if pos_scale < 0.0 or pos_scale > 1.0:
                print(f"警告: 位置比例 {pos_scale} 超出范围 [0.0, 1.0]，进行钳位")
                pos_scale = max(0.0, min(1.0, pos_scale))
            
            lower = self.config.dof_lower[joint_idx]
            upper = self.config.dof_upper[joint_idx]
            joint_state.target_position = lower + pos_scale * (upper - lower)
            joint_state.anim_state = ANIM_SEEK_SCALE
            
            print(f"设置关节 {joint_idx} 位置比例为 {pos_scale:.2f}, 目标位置: {joint_state.target_position:.3f} rad")
        
        if speed_scale is not None:
            original_values['speed'] = joint_state.speed
            range_size = self.config.dof_upper[joint_idx] - self.config.dof_lower[joint_idx]
            joint_state.speed = speed_scale * min(range_size, 3.0 * math.pi)
            print(f"设置关节 {joint_idx} 速度比例为 {speed_scale:.2f}, 新速度: {joint_state.speed:.3f} rad/s")
        
        if kp is not None or kd is not None:
            original_values['kp'] = self.config.kps[joint_idx]
            original_values['kd'] = self.config.kds[joint_idx]
            
            if kp is not None:
                self.config.kps[joint_idx] = kp
            if kd is not None:
                self.config.kds[joint_idx] = kd
                
            print(f"设置关节 {joint_idx} 增益: KP={kp if kp is not None else self.config.kps[joint_idx]:.2f}, "
                  f"KD={kd if kd is not None else self.config.kds[joint_idx]:.2f}")
        
        if min_range is not None and max_range is not None:
            original_values['range'] = (self.config.dof_lower[joint_idx], self.config.dof_upper[joint_idx])
            
            min_range = max(min_range, self.config.dof_lower[joint_idx])
            max_range = min(max_range, self.config.dof_upper[joint_idx])
            
            self.config.dof_lower[joint_idx] = min_range
            self.config.dof_upper[joint_idx] = max_range
            print(f"设置关节 {joint_idx} 范围: [{min_range:.3f}, {max_range:.3f}] rad")
        
        joint_state.original_settings[joint_idx] = original_values
        return original_values
    
    def reset_joint_original(self, joint_idx: int) -> Dict[str, Any]:
        """
        重置关节参数为原始值
        
        Args:
            joint_idx: 要重置的关节索引
            
        Returns:
            被重置的参数字典
        """
        joint_state = self.joint_states[joint_idx]
        
        if joint_idx not in joint_state.original_settings:
            print(f"关节 {joint_idx} 没有记录的原始设置")
            return {}
        
        original_values = joint_state.original_settings[joint_idx]
        
        if 'speed' in original_values:
            joint_state.speed = original_values['speed']
            print(f"重置关节 {joint_idx} 速度为原始值: {joint_state.speed:.3f} rad/s")
        
        if 'kp' in original_values:
            self.config.kps[joint_idx] = original_values['kp']
        if 'kd' in original_values:
            self.config.kds[joint_idx] = original_values['kd']
            print(f"重置关节 {joint_idx} 增益为原始值: KP={self.config.kps[joint_idx]:.2f}, KD={self.config.kds[joint_idx]:.2f}")
        
        if 'range' in original_values:
            orig_min, orig_max = original_values['range']
            self.config.dof_lower[joint_idx] = orig_min
            self.config.dof_upper[joint_idx] = orig_max
            print(f"重置关节 {joint_idx} 范围为原始值: [{orig_min:.3f}, {orig_max:.3f}] rad")
        
        return original_values
    
    def set_joint_scale(self, joint_idx: int, scale: float):
        """设置关节位置比例 (0.0 到 1.0)"""
        return self.configure_joint(joint_idx, pos_scale=scale)
    
    def set_joint_speed(self, joint_idx: int, speed_scale: float):
        """设置关节速度比例"""
        return self.configure_joint(joint_idx, speed_scale=speed_scale)
    
    def set_joint_gains(self, joint_idx: int, kp: Optional[float] = None, kd: Optional[float] = None):
        """设置关节PID增益"""
        return self.configure_joint(joint_idx, kp=kp, kd=kd)
    
    def set_joint_range(self, joint_idx: int, min_range: float, max_range: float):
        """设置关节范围限制"""
        return self.configure_joint(joint_idx, min_range=min_range, max_range=max_range)
    
    def compute_torque(self, position_error: float, dof_velocity: float, joint_idx: int) -> float:
        """使用PD控制计算扭矩"""
        kp = self.config.kps[joint_idx]
        kd = self.config.kds[joint_idx]
        return kp * position_error - kd * dof_velocity
    
    def safe_joint_control(self, joint_idx: int, desired_position: float) -> float:
        """
        安全的关节控制，包含边界检查和异常处理
        
        Args:
            joint_idx: 关节索引
            desired_position: 期望位置
            
        Returns:
            安全的位置值
        """
        try:
            # 边界检查
            if joint_idx < 0 or joint_idx >= self.config.num_joints:
                raise ValueError(f"无效的关节索引: {joint_idx}")
            
            # 钳位到关节限制
            clamped_pos = np.clip(
                desired_position, 
                self.config.dof_lower[joint_idx],
                self.config.dof_upper[joint_idx]
            )
            
            return clamped_pos
            
        except Exception as e:
            print(f"关节 {joint_idx} 控制错误: {e}")
            # 返回当前位置或默认位置保证安全
            return self.config.default_joint_pos[joint_idx]
    
    # def calculate_initial_phase(self, current_pos: float, center: float, amplitude: float, 
    #                       lower_limit: float, upper_limit: float) -> float:
    #     """
    #     根据当前位置计算正弦运动的初始相位（完整周期）
        
    #     Args:
    #         current_pos: 关节当前位置
    #         center: 正弦运动中心位置
    #         amplitude: 正弦运动幅度
            
    #     Returns:
    #         初始相位 (弧度)
    #     """
    #     if abs(amplitude) < 1e-6:
    #         return 0.0
        
    #     # 计算归一化位置
    #     normalized_pos = (current_pos - center) / amplitude
    #     normalized_pos = np.clip(normalized_pos, -1.0, 1.0)
        
    #     # 方法1：使用arcsin，但考虑完整周期
    #     phase = np.arcsin(normalized_pos)
        
    #     # 检查是否需要在第二象限
    #     # 如果当前位置在中心之上且相位为负，调整到第二象限
    #     if current_pos > center and phase < 0:
    #         phase = np.pi - phase
    #     # 如果当前位置在中心之下且相位为正，调整到第三象限
    #     elif current_pos < center and phase > 0:
    #         phase = np.pi - phase
        
    #     return phase

    def calculate_initial_phase(self, current_pos: float, center: float, amplitude: float, 
                          lower_limit: float, upper_limit: float) -> float:
        """
        根据当前位置计算正弦运动的初始相位
        确保从当前位置平滑开始，并根据离上下限距离选择运动方向
        
        Args:
            current_pos: 关节当前位置
            center: 正弦运动中心位置
            amplitude: 正弦运动幅度
            lower_limit: 关节下限
            upper_limit: 关节上限
            
        Returns:
            初始相位 (弧度)
        """
        if abs(amplitude) < 1e-6:
            return 0.0
        
        # 计算当前位置离上下限的距离
        dist_to_upper = upper_limit - current_pos
        dist_to_lower = current_pos - lower_limit
        
        # 计算归一化位置
        normalized_pos = (current_pos - center) / amplitude
        normalized_pos = np.clip(normalized_pos, -1.0, 1.0)
        
        # 基础相位
        base_phase = np.arcsin(normalized_pos)
        
        # 根据离上下限的距离决定运动方向
        if dist_to_upper < dist_to_lower:
            # 离上限更近，应该向上限运动
            # 在正弦波中，向上运动对应相位在 [-π/2, π/2] 区间，且cos(phase)>0
            if np.cos(base_phase) < 0:
                # 调整相位使速度方向为正（向上）
                phase = np.pi - base_phase
            else:
                phase = base_phase
        else:
            # 离下限更近，应该向下限运动  
            # 在正弦波中，向下运动对应相位在 [π/2, 3π/2] 区间，且cos(phase)<0
            if np.cos(base_phase) > 0:
                # 调整相位使速度方向为负（向下）
                phase = np.pi - base_phase
            else:
                phase = base_phase
        
        # 确保相位在 [0, 2π) 范围内
        phase = phase % (2 * np.pi)
        
        # # 调试信息
        print(f"初始相位计算: 位置={current_pos:.3f}, 中心={center:.3f}, 幅度={amplitude:.3f}")
        print(f"  离上限距离={dist_to_upper:.3f}, 离下限距离={dist_to_lower:.3f}")
        print(f"  基础相位={base_phase:.3f}, 最终相位={phase:.3f}")
        print(f"  初始速度方向={'向上' if np.cos(phase) > 0 else '向下'}")
        
        return phase
    
    def cmd_joint_single(self, dof_pos: np.ndarray, joint_idx: int, 
                        mode: Optional[int] = None) -> Tuple[np.ndarray, float]:
        """
        单关节控制命令
        
        Args:
            dof_pos: 当前关节位置数组
            joint_idx: 目标关节索引
            mode: 运动模式，None表示使用关节的当前状态
            
        Returns:
            (新的关节位置, 关节的delta变化)
        """
        joint_state = self.joint_states[joint_idx]
        
        if mode is not None:
            joint_state.anim_state = mode
        
        current_mode = joint_state.anim_state
        current_pos = dof_pos[joint_idx]
        dof_delta = 0.0
        
        if current_mode == ANIM_SEEK_LOWER:
            # 移动到下限
            target_pos = self.config.dof_lower[joint_idx]
            error = current_pos - target_pos
            # 计算平滑速度曲线
            if error < 0.1:  # 接近下限时减速
                # 使用平滑的减速曲线 (二次函数)
                t = error / 0.1  # 归一化到 [0,1]
                speed_factor = t * t  # 二次减速
                delta = -joint_state.speed * self.config.control_dt * speed_factor
            else:
                delta = -joint_state.speed * self.config.control_dt
            
            delta = -joint_state.speed * self.config.control_dt


            new_pos = current_pos + delta
            dof_delta = delta
            
            if new_pos <= self.config.dof_lower[joint_idx]:
                new_pos = self.config.dof_lower[joint_idx]
                joint_state.anim_state = ANIM_SEEK_UPPER
            
            dof_pos[joint_idx] = new_pos
            
        elif current_mode == ANIM_SEEK_UPPER:
            # 移动到上限
            target_pos = self.config.dof_upper[joint_idx]
            error = target_pos - current_pos
            
            # 计算平滑速度曲线
            if error < 0.1:  # 接近上限时减速
                # 使用平滑的减速曲线 (二次函数)
                t = error / 0.1  # 归一化到 [0,1]
                speed_factor = t * t  # 二次减速
                delta = joint_state.speed * self.config.control_dt * speed_factor
            else:
                delta = joint_state.speed * self.config.control_dt
            new_pos = current_pos + delta
            dof_delta = delta
            
            if new_pos >= self.config.dof_upper[joint_idx]:
                new_pos = self.config.dof_upper[joint_idx]
                joint_state.anim_state = ANIM_SEEK_LOWER
            
            dof_pos[joint_idx] = new_pos
            
        elif current_mode == ANIM_SEEK_DEFAULT:
            # 移动到默认位置
            delta = -joint_state.speed * self.config.control_dt
            new_pos = current_pos + delta
            dof_delta = delta
            
            if new_pos <= self.config.default_joint_pos[joint_idx]:
                new_pos = self.config.default_joint_pos[joint_idx]
                joint_state.anim_state = ANIM_FINISHED
            
            dof_pos[joint_idx] = new_pos

        elif current_mode == ANIM_SINE_CUSTOM:
            if not hasattr(joint_state, 'sine_params'):
                lower = self.config.dof_lower[joint_idx]
                upper = self.config.dof_upper[joint_idx]
                center = (upper + lower) / 2
                amplitude = min((upper - lower)/2, center - lower, upper - center)
                calculated_frequency = joint_state.speed / (2 * np.pi * amplitude)
                
                # 获取当前状态
                current_pos = dof_pos[joint_idx]
                current_vel = getattr(joint_state, 'current_velocity', 0.0)  # 如果有速度信息
                
                # 计算初始相位
                initial_phase = self.calculate_initial_phase(current_pos, center, amplitude,self.config.dof_lower[joint_idx], self.config.dof_upper[joint_idx])
                
                joint_state.sine_params = {
                    'amplitude': amplitude,
                    'frequency': calculated_frequency,
                    'center': center,
                    'phase': initial_phase,
                    'smooth_start': True,  # 平滑启动标志
                    'start_time': 0.0
                }
            
            calculated_frequency = joint_state.speed / (2 * np.pi *  joint_state.sine_params['amplitude'])
            joint_state.sine_params['frequency'] = calculated_frequency
            params = joint_state.sine_params
            
            # 平滑启动：前0.2秒内限制最大速度
            if params['smooth_start'] and params['start_time'] < 0.2:
                params['start_time'] += self.config.control_dt
                
                # 限制最大速度变化
                max_velocity = params['amplitude'] * 2 * np.pi * params['frequency']
                limited_velocity = max_velocity * (params['start_time'] / 0.2)  # 线性加速
                
                # 使用限制后的速度更新相位
                effective_frequency = limited_velocity / (2 * np.pi * params['amplitude'])
                params['phase'] += 2 * np.pi * effective_frequency * self.config.control_dt
            else:
                params['smooth_start'] = False
                params['phase'] += 2 * np.pi * params['frequency'] * self.config.control_dt
            
            # 计算目标位置
            target_pos = params['center'] + params['amplitude'] * np.sin(params['phase'])
            new_pos = np.clip(target_pos, self.config.dof_lower[joint_idx], self.config.dof_upper[joint_idx])
            dof_delta = new_pos - current_pos
            dof_pos[joint_idx] = new_pos
                    
        elif current_mode == ANIM_SEEK_SCALE:
            # 移动到比例位置
            if joint_state.target_position is None:
                joint_state.target_position = self.config.default_joint_pos[joint_idx]
            
            target_pos = joint_state.target_position
            error = target_pos - current_pos
            
            max_speed = joint_state.speed
            min_speed = 0.1
            
            # 基于误差计算自适应速度
            speed = min_speed + (max_speed - min_speed) * min(abs(error) / 0.5, 1.0)
            
            if abs(error) > 0.05:
                if error > 0:
                    # 向目标移动
                    delta = speed * self.config.control_dt
                    new_pos = current_pos + delta
                    dof_delta = delta
                else:
                    # 向目标移动
                    delta = -speed * self.config.control_dt
                    new_pos = current_pos + delta
                    dof_delta = delta
                
                # 钳位位置到关节限制
                new_pos = self.safe_joint_control(joint_idx, new_pos)
                dof_pos[joint_idx] = new_pos
            else:
                # 到达目标位置
                dof_delta = 0
                dof_pos[joint_idx] = target_pos
        
        elif current_mode == ANIM_S_CURVE:
            # S曲线运动
            if not hasattr(joint_state, 'scurve_params'):
                lower = self.config.dof_lower[joint_idx]
                upper = self.config.dof_upper[joint_idx]
                
                joint_state.scurve_params = {
                    'amplitude': (upper - lower) * 0.4,
                    'frequency': 0.2,
                    'center': (upper + lower) / 2,
                    'phase': 0.0,
                    'max_velocity': 1.5,
                    'max_acceleration': 3.0,
                    'max_jerk': 10.0,
                    'current_velocity': 0.0,
                    'current_acceleration': 0.0,
                    'direction': 1,
                    'segment': 'jerk_up'  # jerk_up, accel_constant, jerk_down, vel_constant, jerk_neg, decel_constant, jerk_zero
                }
            
            params = joint_state.scurve_params
            amplitude = params['amplitude']
            center = params['center']
            max_vel = params['max_velocity']
            max_accel = params['max_acceleration']
            max_jerk = params['max_jerk']
            
            # S曲线七段式状态机
            if params['segment'] == 'jerk_up':
                # 加加速度阶段
                params['current_acceleration'] += max_jerk * self.config.control_dt * params['direction']
                if abs(params['current_acceleration']) >= max_accel:
                    params['current_acceleration'] = max_accel * params['direction']
                    params['segment'] = 'accel_constant'
            
            elif params['segment'] == 'accel_constant':
                # 匀加速度阶段
                # 检查是否需要开始减加速度
                if abs(params['current_velocity']) >= max_vel / 2:
                    params['segment'] = 'jerk_down'
            
            elif params['segment'] == 'jerk_down':
                # 减加速度阶段
                params['current_acceleration'] -= max_jerk * self.config.control_dt * params['direction']
                if abs(params['current_acceleration']) <= 0.01:
                    params['current_acceleration'] = 0.0
                    params['segment'] = 'vel_constant'
            
            elif params['segment'] == 'vel_constant':
                # 匀速阶段
                # 检查是否需要开始减速
                target_pos = center + amplitude * params['direction']
                remaining_distance = abs(target_pos - current_pos)
                
                # 计算减速所需距离（对称于加速阶段）
                decel_distance = (max_vel**2) / (2 * max_accel)
                if remaining_distance <= decel_distance:
                    params['segment'] = 'jerk_neg'
            
            elif params['segment'] == 'jerk_neg':
                # 负加加速度阶段（开始减速）
                params['current_acceleration'] -= max_jerk * self.config.control_dt * params['direction']
                if abs(params['current_acceleration']) >= -max_accel:
                    params['current_acceleration'] = -max_accel * params['direction']
                    params['segment'] = 'decel_constant'
            
            elif params['segment'] == 'decel_constant':
                # 匀减速度阶段
                if abs(params['current_velocity']) <= 0.1:
                    params['segment'] = 'jerk_zero'
            
            elif params['segment'] == 'jerk_zero':
                # 减加速度到零阶段
                params['current_acceleration'] += max_jerk * self.config.control_dt * params['direction']
                if abs(params['current_acceleration']) <= 0.01:
                    params['current_acceleration'] = 0.0
                    params['current_velocity'] = 0.0
                    params['direction'] *= -1
                    params['segment'] = 'jerk_up'
            
            # 更新速度和位置
            params['current_velocity'] += params['current_acceleration'] * self.config.control_dt
            params['current_velocity'] = np.clip(params['current_velocity'], -max_vel, max_vel)
            
            new_pos = current_pos + params['current_velocity'] * self.config.control_dt
            
            # 限制在关节范围内
            new_pos = np.clip(new_pos, self.config.dof_lower[joint_idx], self.config.dof_upper[joint_idx])
            dof_delta = new_pos - current_pos
            dof_pos[joint_idx] = new_pos
        
        elif current_mode == ANIM_TRAPEZOIDAL_VEL:
            # 梯形速度运动
            if not hasattr(joint_state, 'trapz_params'):
                lower = self.config.dof_lower[joint_idx]
                upper = self.config.dof_upper[joint_idx]
                
                joint_state.trapz_params = {
                    'amplitude': (upper - lower) * 0.4,
                    'frequency': 0.3,
                    'center': (upper + lower) / 2,
                    'phase': 0.0,
                    'max_velocity': 2.0,  # 最大速度
                    'acceleration': 5.0,   # 加速度
                    'current_velocity': 0.0,
                    'direction': 1,
                    'state': 'accelerating'  # accelerating, constant, decelerating
                }
            
            params = joint_state.trapz_params
            amplitude = params['amplitude']
            center = params['center']
            max_vel = params['max_velocity']
            accel = params['acceleration']
            
            # 根据状态更新速度
            if params['state'] == 'accelerating':
                params['current_velocity'] += accel * self.config.control_dt * params['direction']
                if abs(params['current_velocity']) >= max_vel:
                    params['current_velocity'] = max_vel * params['direction']
                    params['state'] = 'constant'
            
            elif params['state'] == 'constant':
                # 检查是否需要开始减速
                remaining_distance = abs((center + amplitude * params['direction']) - current_pos)
                stop_distance = (params['current_velocity']**2) / (2 * accel)
                
                if remaining_distance <= stop_distance:
                    params['state'] = 'decelerating'
            
            elif params['state'] == 'decelerating':
                params['current_velocity'] -= accel * self.config.control_dt * params['direction']
                
                # 检查是否到达目标并反转方向
                target_pos = center + amplitude * params['direction']
                if abs(params['current_velocity']) < 0.01 or (
                    (params['direction'] == 1 and current_pos >= target_pos) or
                    (params['direction'] == -1 and current_pos <= target_pos)
                ):
                    params['current_velocity'] = 0.0
                    params['direction'] *= -1
                    params['state'] = 'accelerating'
            
            # 更新位置
            new_pos = current_pos + params['current_velocity'] * self.config.control_dt
            
            # 限制在关节范围内
            new_pos = np.clip(new_pos, self.config.dof_lower[joint_idx], self.config.dof_upper[joint_idx])
            dof_delta = new_pos - current_pos
            dof_pos[joint_idx] = new_pos
        
        elif current_mode == ANIM_TRIANGLE_WAVE:
            # 三角波运动
            if not hasattr(joint_state, 'triangle_params'):
                lower = self.config.dof_lower[joint_idx]
                upper = self.config.dof_upper[joint_idx]
                
                joint_state.triangle_params = {
                    'amplitude': (upper - lower) * 0.3,
                    'frequency': 0.5,
                    'center': (upper + lower) / 2,
                    'phase': 0.0,
                    'direction': 1  # 1表示上升，-1表示下降
                }
            
            params = joint_state.triangle_params
            amplitude = params['amplitude']
            frequency = params['frequency']
            center = params['center']
            
            # 更新相位
            phase_increment = 4 * amplitude * frequency * self.config.control_dt
            params['phase'] += phase_increment * params['direction']
            
            # 计算三角波位置
            if params['direction'] == 1:
                # 上升阶段
                target_pos = center - amplitude + params['phase']
                if target_pos >= center + amplitude:
                    target_pos = center + amplitude
                    params['direction'] = -1
                    params['phase'] = 2 * amplitude  # 重置相位
            else:
                # 下降阶段
                target_pos = center + amplitude - params['phase']
                if target_pos <= center - amplitude:
                    target_pos = center - amplitude
                    params['direction'] = 1
                    params['phase'] = 2 * amplitude  # 重置相位
            
            # 限制在关节范围内
            new_pos = np.clip(target_pos, self.config.dof_lower[joint_idx], self.config.dof_upper[joint_idx])
            dof_delta = new_pos - current_pos
            dof_pos[joint_idx] = new_pos
        
        elif current_mode == ANIM_SQUARE_WAVE:
            # 方波运动
            if not hasattr(joint_state, 'square_params'):
                lower = self.config.dof_lower[joint_idx]
                upper = self.config.dof_upper[joint_idx]
                
                joint_state.square_params = {
                    'low_pos': lower + (upper - lower) * 0.2,
                    'high_pos': upper - (upper - lower) * 0.2,
                    'frequency': 0.5,
                    'phase': 0.0,
                    'current_target': None,
                    'transition_speed': 2.0  # 过渡速度，避免瞬时切换
                }
            
            params = joint_state.square_params
            low_pos = params['low_pos']
            high_pos = params['high_pos']
            frequency = params['frequency']
            transition_speed = params['transition_speed']
            
            # 初始化当前目标
            if params['current_target'] is None:
                params['current_target'] = low_pos if current_pos < (low_pos + high_pos) / 2 else high_pos
            
            # 更新相位
            params['phase'] += frequency * self.config.control_dt
            
            # 检查是否需要切换目标
            if params['phase'] >= 1.0:
                params['phase'] = 0.0
                # 切换目标位置
                params['current_target'] = high_pos if params['current_target'] == low_pos else low_pos
            
            # 向目标位置移动
            error = params['current_target'] - current_pos
            max_step = transition_speed * self.config.control_dt
            
            if abs(error) > max_step:
                # 需要多步移动
                new_pos = current_pos + np.sign(error) * max_step
            else:
                # 到达目标位置
                new_pos = params['current_target']
            
            # 限制在关节范围内
            new_pos = np.clip(new_pos, self.config.dof_lower[joint_idx], self.config.dof_upper[joint_idx])
            dof_delta = new_pos - current_pos
            dof_pos[joint_idx] = new_pos
        
        return dof_pos, dof_delta
    
    def cmd_joints(self, dof_pos: np.ndarray, target_joints: List[int] = None, 
                  modes: Dict[int, int] = None) -> Tuple[np.ndarray, List[float]]:
        """
        统一的关节控制函数
        
        Args:
            dof_pos: 当前关节位置数组
            target_joints: 目标关节列表，None表示所有关节
            modes: 每个关节的运动模式字典 {joint_idx: anim_state}
            
        Returns:
            (新的关节位置, 各关节的delta列表)
        """
        if target_joints is None:
            target_joints = list(range(self.config.num_joints))
        
        if modes is None:
            # 默认所有关节使用全局状态
            modes = {joint_idx: self.global_anim_state for joint_idx in target_joints}
        
        for joint_idx in range(self.config.num_joints):
            if joint_idx not in target_joints:
                dof_pos[joint_idx] = self.config.default_joint_pos[joint_idx]
        
        dof_delta = [0.0] * self.config.num_joints
        
        for joint_idx in target_joints:
            mode = modes.get(joint_idx, self.global_anim_state)
            dof_pos, joint_delta = self.cmd_joint_single(dof_pos, joint_idx, mode)
            dof_delta[joint_idx] = joint_delta
        
        return dof_pos, dof_delta
    
    def start_performance_monitor(self):
        """启动性能监控"""
        self.performance_data = {
            'control_times': deque(maxlen=100),
            'last_control_time': time.time(),
            'start_time': time.time()
        }
    
    def log_control_time(self):
        """记录控制周期时间"""
        current_time = time.time()
        if hasattr(self, 'performance_data') and 'last_control_time' in self.performance_data:
            control_time = current_time - self.performance_data['last_control_time']
            self.performance_data['control_times'].append(control_time)
        self.performance_data['last_control_time'] = current_time
    
    def get_performance_stats(self) -> Dict[str, float]:
        """获取性能统计"""
        if not hasattr(self, 'performance_data') or not self.performance_data['control_times']:
            return {}
        
        times = list(self.performance_data['control_times'])
        avg_time = np.mean(times)
        
        return {
            'avg_control_time': avg_time,
            'max_control_time': np.max(times) if times else 0,
            'min_control_time': np.min(times) if times else 0,
            'control_frequency': 1.0 / avg_time if avg_time > 0 else 0,
            'total_runtime': time.time() - self.performance_data.get('start_time', self.start_time)
        }
    
    
    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        # while self.remote_controller.button[KeyMap.start] != 1:
        iii = 0
        while True:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.DT)
            iii += 0.02
            if iii > 5: break
    
    def move_to_default_pos(self):
        print("移动到默认位置...")
        if self.preview:
            self.e1_sim.move_to_default_pos()
        else:
            self.e1_real.move_to_default_pos()
    
    def damping_mode(self):
        """进入阻尼模式"""
        print("进入阻尼模式...")
        self.e1_real.damping_mode()
    
    def run(self):
        """主运行循环（由子类实现）"""
        raise NotImplementedError("子类必须实现run方法")


if __name__ == "__main__":
    # 示例用法
    ChannelFactoryInitialize(0)

    config_path = "test/o1.yaml"
    config = Config(config_path)
    controller = BaseController(config, "test", preview=False, visualization=True)
    
    try:
        # 启动性能监控
        controller.start_performance_monitor()
        
        # 运行控制器
        step_count = 0
        while True:
            # 记录控制时间
            controller.log_control_time()
            
            # 运行控制逻辑
            controller.run()
            step_count += 1
            
            # 更新可视化
            controller.update_visualization(step_count)
            
    except KeyboardInterrupt:
        print("程序被用户中断")
    finally:
        # 保存数据
        controller.save_all_data()
        print("程序结束")
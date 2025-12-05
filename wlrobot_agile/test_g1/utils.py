import threading
import queue
import csv
import time
from typing import List, Dict
from config import Config
import matplotlib.pyplot as plt

class DataSaverThread(threading.Thread):
    """数据保存线程"""
    
    def __init__(self, data_file: str, config: Config):
        super().__init__()
        self.data_file = data_file
        self.config = config
        self.data_queue = queue.Queue()
        self.running = True
        self.daemon = True
        
        # 初始化CSV文件
        with open(self.data_file, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['time', 'joint_idx', 'target_pos', 'actual_pos', 'target_vel', 'actual_vel', 
                     'compute_torque', 'actual_torque', 'kp', 'kd', 'speed', 'error']
            writer.writerow(header)
    
    def add_data(self, data_row: List):
        """向队列添加数据"""
        self.data_queue.put(data_row)
    
    def run(self):
        """线程主循环"""
        while self.running:
            try:
                # 批量处理数据，减少文件IO次数
                data_batch = []
                while len(data_batch) < 10:  # 最多批量处理10条数据
                    try:
                        data_row = self.data_queue.get(timeout=0.1)
                        data_batch.append(data_row)
                    except queue.Empty:
                        break
                
                if data_batch:
                    # 批量写入文件
                    with open(self.data_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerows(data_batch)
                
            except Exception as e:
                print(f"数据保存线程错误: {e}")
    
    def stop(self):
        """停止线程"""
        self.running = False
        # 处理剩余数据
        while not self.data_queue.empty():
            try:
                data_row = self.data_queue.get_nowait()
                with open(self.data_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(data_row)
            except queue.Empty:
                break


class VisualizationThread(threading.Thread):
    """可视化线程"""
    
    def __init__(self, config: Config, joint_data: Dict, data_lock: threading.Lock):
        super().__init__()
        self.config = config
        self.joint_data = joint_data
        self.data_lock = data_lock
        self.update_queue = queue.Queue()
        self.running = True
        self.daemon = True
        self.visualization_interval = 10
        
        # 初始化Matplotlib
        plt.ion()
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 初始化图表
        self.setup_visualization()
    
    def setup_visualization(self):
        """设置可视化图表"""
        # 位置跟踪图表
        self.target_lines = []
        self.actual_lines = []
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        
        for i in range(min(self.config.num_joints, len(colors))):
            target_line, = self.ax1.plot([], [], f'{colors[i]}-', 
                                       label=f'Joint {i} Target', linewidth=1, alpha=0.7)
            actual_line, = self.ax1.plot([], [], f'{colors[i]}--', 
                                      label=f'Joint {i} Actual', linewidth=1, alpha=0.7)
            self.target_lines.append(target_line)
            self.actual_lines.append(actual_line)
        
        # 速度跟踪图表
        self.target_vel_lines = []
        self.actual_vel_lines = []
        for i in range(min(self.config.num_joints, len(colors))):
            target_vel_line, = self.ax2.plot([], [], f'{colors[i]}-', 
                                           label=f'Joint {i} Target', linewidth=1, alpha=0.7)
            actual_vel_line, = self.ax2.plot([], [], f'{colors[i]}--', 
                                          label=f'Joint {i} Actual', linewidth=1, alpha=0.7)
            self.target_vel_lines.append(target_vel_line)
            self.actual_vel_lines.append(actual_vel_line)
        
        # 扭矩图表
        self.compute_torque_lines = []
        self.actual_torque_lines = []
        for i in range(min(self.config.num_joints, len(colors))):
            compute_torque_line, = self.ax3.plot([], [], f'{colors[i]}-', 
                                               label=f'Joint {i} Computed', linewidth=1, alpha=0.7)
            actual_torque_line, = self.ax3.plot([], [], f'{colors[i]}--', 
                                              label=f'Joint {i} Actual', linewidth=1, alpha=0.7)
            self.compute_torque_lines.append(compute_torque_line)
            self.actual_torque_lines.append(actual_torque_line)
            
        # 误差图表
        self.error_lines = []
        for i in range(min(self.config.num_joints, len(colors))):
            error_line, = self.ax4.plot([], [], f'{colors[i]}-', 
                                      label=f'Joint {i} Error', linewidth=1, alpha=0.7)
            self.error_lines.append(error_line)
        
        # 设置图表属性
        self.ax1.set_xlabel('Time (s)')
        self.ax1.set_ylabel('Position (rad)')
        self.ax1.set_title('Real-time Position Tracking')
        self.ax1.legend()
        self.ax1.grid(True)
        
        self.ax2.set_xlabel('Time (s)')
        self.ax2.set_ylabel('Velocity (rad/s)')
        self.ax2.set_title('Real-time Velocity Tracking')
        self.ax2.legend()
        self.ax2.grid(True)
        
        self.ax3.set_xlabel('Time (s)')
        self.ax3.set_ylabel('Torque (Nm)')
        self.ax3.set_title('Real-time Torque')
        self.ax3.legend()
        self.ax3.grid(True)
        
        self.ax4.set_xlabel('Time (s)')
        self.ax4.set_ylabel('Error (rad)')
        self.ax4.set_title('Position Error')
        self.ax4.legend()
        self.ax4.grid(True)
        
        # 性能文本显示
        self.performance_text = self.ax1.text(
            0.02, 0.95, '', transform=self.ax1.transAxes, 
            bbox=dict(facecolor='white', alpha=0.7)
        )
        
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
    
    def request_update(self, step_count: int, performance_stats: Dict):
        """请求更新可视化"""
        if step_count % self.visualization_interval == 0:
            self.update_queue.put((step_count, performance_stats))
    
    def run(self):
        """线程主循环"""
        while self.running:
            try:
                # 等待更新请求
                step_count, performance_stats = self.update_queue.get(timeout=0.5)
                self._update_visualization(step_count, performance_stats)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"可视化线程错误: {e}")
    
    def _update_visualization(self, step_count: int, performance_stats: Dict):
        """更新可视化图表（在线程中执行）"""
        # 线程安全的数据检索
        with self.data_lock:
            joint_data_copy = {}
            for joint_idx, data in self.joint_data.items():
                joint_data_copy[joint_idx] = {
                    key: value.copy() for key, value in data.items()
                }
        
        if not joint_data_copy:
            return
        
        # 更新所有图表
        max_time = 0
        for joint_idx, data in joint_data_copy.items():
            if data['time']:
                max_time = max(max_time, max(data['time']))
                
                # 只显示前几个关节以避免图表过于拥挤
                if joint_idx < len(self.target_lines):
                    # 更新位置图表
                    self.target_lines[joint_idx].set_data(data['time'], data['target_pos'])
                    self.actual_lines[joint_idx].set_data(data['time'], data['actual_pos'])
                    
                    # 更新速度图表
                    self.target_vel_lines[joint_idx].set_data(data['time'], data['target_vel'])
                    self.actual_vel_lines[joint_idx].set_data(data['time'], data['actual_vel'])
                    
                    # 更新扭矩图表
                    self.compute_torque_lines[joint_idx].set_data(data['time'], data['compute_torque'])
                    self.actual_torque_lines[joint_idx].set_data(data['time'], data['actual_torque'])
                    
                    # 更新误差图表
                    self.error_lines[joint_idx].set_data(data['time'], data['error'])
        
        # 调整坐标轴范围
        if max_time > 0:
            for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
                ax.set_xlim(0, max(10, max_time))
        
        # 更新性能显示
        perf_text = f"Control Freq: {performance_stats.get('control_frequency', 0):.1f} Hz\n"
        perf_text += f"Avg Ctrl Time: {performance_stats.get('avg_control_time', 0)*1000:.1f} ms"
        self.performance_text.set_text(perf_text)
        
        # 更新图表
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def stop(self):
        """停止线程"""
        self.running = False
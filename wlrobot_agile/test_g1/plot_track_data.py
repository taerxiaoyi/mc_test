import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.fft import fft, fftfreq
import os
import glob
from typing import Dict, List, Tuple, Optional
import json

class RobotDataAnalyzer:
    """Robot Data Analyzer - Save separate charts for each joint"""
    
    def __init__(self, data_file: str):
        """
        Initialize data analyzer
        
        Args:
            data_file: Path to data file
        """
        self.data_file = data_file
        self.data_dir = os.path.dirname(data_file)
        
        # Joint names mapping
        self.joint_names = [
            'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
            'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
            'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
            'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
            'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint'
        ]
        
        # Set font to avoid Chinese character issues
        plt.rcParams['font.family'] = 'DejaVu Sans'  # Use a common font that supports more characters
        plt.rcParams['axes.unicode_minus'] = False  # Properly display minus signs
        
        # Load data
        self.df = self.load_data()
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        print(f"Loading data: {self.data_file}")
        print(f"Time range: {self.df['time'].min():.2f} - {self.df['time'].max():.2f} seconds")
        print(f"Number of joints: {self.df['joint_idx'].nunique()}")
        print(f"Number of joint names: {len(self.joint_names)}")
    
    def load_data(self) -> pd.DataFrame:
        """Load CSV data"""
        df = pd.read_csv(self.data_file)
        
        # Data cleaning
        df = df.dropna()  # Remove NaN values
        df = df[df['time'] >= 0]  # Remove negative time
        
        # Calculate additional metrics
        df['abs_error'] = np.abs(df['error'])
        df['squared_error'] = df['error'] ** 2
        
        return df
    
    def get_joint_name(self, joint_idx: int) -> str:
        """Get joint name from joint index"""
        if 0 <= joint_idx < len(self.joint_names):
            return self.joint_names[joint_idx]
        else:
            return f"joint_{joint_idx}"
    
    def get_joint_data(self, joint_idx: int) -> pd.DataFrame:
        """Get data for specific joint"""
        return self.df[self.df['joint_idx'] == joint_idx].copy()
    
    def get_all_joints_data(self) -> Dict[int, pd.DataFrame]:
        """Get data dictionary for all joints"""
        joints = self.df['joint_idx'].unique()
        return {joint: self.get_joint_data(joint) for joint in joints}
    
    def calculate_joint_statistics(self, joint_idx: int) -> Dict:
        """Calculate statistics for single joint"""
        joint_data = self.get_joint_data(joint_idx)
        
        if len(joint_data) == 0:
            return {}
        
        stats = {
            'joint_idx': joint_idx,
            'joint_name': self.get_joint_name(joint_idx),
            'data_points': len(joint_data),
            'time_span': joint_data['time'].max() - joint_data['time'].min(),
            'mean_error': joint_data['error'].mean(),
            'std_error': joint_data['error'].std(),
            'max_abs_error': joint_data['abs_error'].max(),
            'rms_error': np.sqrt(joint_data['squared_error'].mean()),
            'mean_target_pos': joint_data['target_pos'].mean(),
            'std_target_pos': joint_data['target_pos'].std(),
            'mean_actual_pos': joint_data['actual_pos'].mean(),
            'std_actual_pos': joint_data['actual_pos'].std(),
            'mean_target_vel': joint_data['target_vel'].mean(),
            'std_target_vel': joint_data['target_vel'].std(),
            'mean_actual_vel': joint_data['actual_vel'].mean(),
            'std_actual_vel': joint_data['actual_vel'].std(),
            'mean_compute_torque': joint_data['compute_torque'].mean(),
            'max_compute_torque': joint_data['compute_torque'].abs().max(),
            'sampling_frequency': len(joint_data) / joint_data['time'].max() if joint_data['time'].max() > 0 else 0
        }
        
        return stats
    
    def plot_single_joint_analysis(self, joint_idx: int, save_dir: str = None):
        """Generate complete analysis chart for single joint"""
        if save_dir is None:
            save_dir = self.data_dir
        
        joint_data = self.get_joint_data(joint_idx)
        joint_name = self.get_joint_name(joint_idx)
        
        if len(joint_data) == 0:
            print(f"Warning: Joint {joint_idx} ({joint_name}) has no data")
            return
        
        # Create figure - 2x3 layout
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Joint Analysis: {joint_name} (Index: {joint_idx})', fontsize=16, y=0.98)
        
        # 1. Position Tracking
        ax = axes[0, 0]
        ax.plot(joint_data['time'], joint_data['target_pos'], 
               label='Target Position', color=self.colors[0], linewidth=2, alpha=0.8)
        ax.plot(joint_data['time'], joint_data['actual_pos'], 
               label='Actual Position', color=self.colors[1], linewidth=2, alpha=0.8)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (rad)')
        ax.set_title('Position Tracking')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Velocity Tracking
        ax = axes[0, 1]
        ax.plot(joint_data['time'], joint_data['target_vel'], 
               label='Target Velocity', color=self.colors[2], linewidth=2, alpha=0.8)
        ax.plot(joint_data['time'], joint_data['actual_vel'], 
               label='Actual Velocity', color=self.colors[3], linewidth=2, alpha=0.8)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity (rad/s)')
        ax.set_title('Velocity Tracking')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Torque Comparison
        ax = axes[0, 2]
        ax.plot(joint_data['time'], joint_data['compute_torque'], 
               label='Computed Torque', color=self.colors[4], linewidth=2, alpha=0.8)
        ax.plot(joint_data['time'], joint_data['actual_torque'], 
               label='Actual Torque', color=self.colors[5], linewidth=2, alpha=0.8)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Torque (Nm)')
        ax.set_title('Torque Output')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Tracking Error
        ax = axes[1, 0]
        ax.plot(joint_data['time'], joint_data['error'], 
               color='red', linewidth=2, alpha=0.8)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Error (rad)')
        ax.set_title('Position Tracking Error')
        ax.grid(True, alpha=0.3)
        
        # 5. Error Distribution Histogram
        ax = axes[1, 1]
        ax.hist(joint_data['error'], bins=50, density=True, alpha=0.7, color='skyblue')
        ax.set_xlabel('Error (rad)')
        ax.set_ylabel('Probability Density')
        ax.set_title('Error Distribution')
        ax.grid(True, alpha=0.3)
        
        # 6. Frequency Domain Analysis
        ax = axes[1, 2]
        
        # Calculate sampling frequency
        time_diff = np.diff(joint_data['time'])
        if len(time_diff) > 0:
            fs = 1.0 / np.mean(time_diff)
        else:
            fs = 100  # Default sampling frequency
        
        # FFT of error signal
        error_signal = joint_data['error'].values
        N = len(error_signal)
        
        if N > 1:
            yf = fft(error_signal)
            xf = fftfreq(N, 1/fs)[:N//2]
            ax.semilogy(xf, 2.0/N * np.abs(yf[0:N//2]), 
                       color='purple', linewidth=2, alpha=0.8)
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Amplitude')
            ax.set_title('Error Spectrum')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Insufficient data\nfor frequency analysis', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Frequency Analysis (Insufficient Data)')
        
        plt.tight_layout()
        
        # Save chart
        safe_joint_name = joint_name.replace(' ', '_').replace('/', '_')
        filename = f"joint_{joint_idx:02d}_{safe_joint_name}.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close figure to free memory
        
        print(f"Saved joint {joint_idx} chart: {filename}")
        
        return save_path
    
    def plot_joint_comparison(self, joint_indices: List[int], save_dir: str = None):
        """Compare performance of multiple joints"""
        if save_dir is None:
            save_dir = self.data_dir
        
        n_joints = len(joint_indices)
        if n_joints == 0:
            return
        
        # Create comparison chart
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Multi-Joint Performance Comparison', fontsize=16)
        
        # 1. RMS Error Comparison
        rms_errors = []
        joint_labels = []
        for joint_idx in joint_indices:
            stats = self.calculate_joint_statistics(joint_idx)
            if stats:
                rms_errors.append(stats['rms_error'])
                joint_labels.append(f"{joint_idx}\n{self.get_joint_name(joint_idx)}")
        
        axes[0, 0].bar(range(len(rms_errors)), rms_errors, color=self.colors[:len(rms_errors)])
        axes[0, 0].set_xticks(range(len(rms_errors)))
        axes[0, 0].set_xticklabels(joint_labels, rotation=45, ha='right')
        axes[0, 0].set_ylabel('RMS Error (rad)')
        axes[0, 0].set_title('RMS Error by Joint')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Maximum Absolute Error Comparison
        max_errors = []
        for joint_idx in joint_indices:
            stats = self.calculate_joint_statistics(joint_idx)
            if stats:
                max_errors.append(stats['max_abs_error'])
        
        axes[0, 1].bar(range(len(max_errors)), max_errors, color=self.colors[:len(max_errors)])
        axes[0, 1].set_xticks(range(len(max_errors)))
        axes[0, 1].set_xticklabels(joint_labels, rotation=45, ha='right')
        axes[0, 1].set_ylabel('Max Absolute Error (rad)')
        axes[0, 1].set_title('Max Absolute Error by Joint')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Position Standard Deviation Comparison
        pos_stds = []
        for joint_idx in joint_indices:
            stats = self.calculate_joint_statistics(joint_idx)
            if stats:
                pos_stds.append(stats['std_actual_pos'])
        
        axes[1, 0].bar(range(len(pos_stds)), pos_stds, color=self.colors[:len(pos_stds)])
        axes[1, 0].set_xticks(range(len(pos_stds)))
        axes[1, 0].set_xticklabels(joint_labels, rotation=45, ha='right')
        axes[1, 0].set_ylabel('Position Std Dev (rad)')
        axes[1, 0].set_title('Position Variation by Joint')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Maximum Torque Comparison
        max_torques = []
        for joint_idx in joint_indices:
            stats = self.calculate_joint_statistics(joint_idx)
            if stats:
                max_torques.append(stats['max_compute_torque'])
        
        axes[1, 1].bar(range(len(max_torques)), max_torques, color=self.colors[:len(max_torques)])
        axes[1, 1].set_xticks(range(len(max_torques)))
        axes[1, 1].set_xticklabels(joint_labels, rotation=45, ha='right')
        axes[1, 1].set_ylabel('Max Torque (Nm)')
        axes[1, 1].set_title('Max Computed Torque by Joint')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save comparison chart
        save_path = os.path.join(save_dir, "joint_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved joint comparison chart: {save_path}")
        
        return save_path
    
    def generate_all_joints_analysis(self, save_dir: str = None):
        """Generate analysis charts for all joints"""
        if save_dir is None:
            save_dir = self.data_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Get all joint indices with data
        all_joints = sorted(self.df['joint_idx'].unique())
        print(f"Starting to generate analysis charts for {len(all_joints)} joints...")
        
        # Generate detailed analysis chart for each joint
        saved_files = []
        for joint_idx in all_joints:
            try:
                save_path = self.plot_single_joint_analysis(joint_idx, save_dir)
                if save_path:
                    saved_files.append(save_path)
            except Exception as e:
                print(f"Error generating chart for joint {joint_idx}: {e}")
        
        # Generate joint comparison chart (select first 12 joints to avoid overcrowding)
        if len(all_joints) > 1:
            try:
                joints_to_compare = all_joints[:min(12, len(all_joints))]
                self.plot_joint_comparison(joints_to_compare, save_dir)
            except Exception as e:
                print(f"Error generating joint comparison chart: {e}")
        
        # Generate statistics report
        stats_report = self.generate_statistics_report(all_joints, save_dir)
        
        print(f"\nAnalysis completed! Generated {len(saved_files)} joint analysis charts")
        print(f"All files saved in: {save_dir}")
        
        return saved_files
    
    def generate_statistics_report(self, joint_indices: List[int], save_dir: str):
        """Generate statistics report"""
        stats_data = []
        
        for joint_idx in joint_indices:
            stats = self.calculate_joint_statistics(joint_idx)
            if stats:
                stats_data.append(stats)
        
        if not stats_data:
            return None
        
        # Convert to DataFrame
        stats_df = pd.DataFrame(stats_data)
        
        # Save as CSV
        csv_path = os.path.join(save_dir, "joint_statistics.csv")
        stats_df.to_csv(csv_path, index=False, float_format='%.6f')
        
        # Generate text report
        report_path = os.path.join(save_dir, "analysis_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Robot Joint Control Performance Analysis Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Data File: {os.path.basename(self.data_file)}\n")
            f.write(f"Analysis Time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of Joints: {len(stats_data)}\n")
            f.write(f"Time Range: {self.df['time'].min():.2f} - {self.df['time'].max():.2f} seconds\n")
            f.write(f"Total Data Points: {len(self.df)}\n\n")
            
            f.write("Overall Performance Metrics:\n")
            f.write(f"  Global RMS Error: {np.sqrt((self.df['error']**2).mean()):.6f} rad\n")
            f.write(f"  Global Max Absolute Error: {self.df['abs_error'].max():.6f} rad\n")
            f.write(f"  Global Mean Error: {self.df['error'].mean():.6f} rad\n")
            f.write(f"  Global Error Std Dev: {self.df['error'].std():.6f} rad\n\n")
            
            f.write("Detailed Statistics by Joint:\n")
            f.write("-" * 60 + "\n")
            
            for stats in stats_data:
                f.write(f"\nJoint {stats['joint_idx']}: {stats['joint_name']}\n")
                f.write(f"  Data Points: {stats['data_points']}\n")
                f.write(f"  Time Span: {stats['time_span']:.2f} s\n")
                f.write(f"  Sampling Frequency: {stats['sampling_frequency']:.1f} Hz\n")
                f.write(f"  RMS Error: {stats['rms_error']:.6f} rad\n")
                f.write(f"  Max Absolute Error: {stats['max_abs_error']:.6f} rad\n")
                f.write(f"  Mean Error: {stats['mean_error']:.6f} rad\n")
                f.write(f"  Error Std Dev: {stats['std_error']:.6f} rad\n")
                f.write(f"  Max Computed Torque: {stats['max_compute_torque']:.3f} Nm\n")
        
        print(f"Statistics report saved to: {report_path}")
        print(f"Statistics data saved to: {csv_path}")
        
        return stats_df

def batch_analyze_folder(folder_path: str):
    """批量分析文件夹中的所有CSV文件"""
    # 查找所有CSV文件
    csv_pattern = os.path.join(folder_path, "**", "*.csv")
    csv_files = glob.glob(csv_pattern, recursive=True)
    
    # 过滤掉不需要的文件
    csv_files = [f for f in csv_files if not any(skip in f for skip in 
                ['_statistics.csv', '_config.json', '.npz', 'joint_'])]
    
    if not csv_files:
        print(f"在 {folder_path} 中未找到CSV文件")
        return
    
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    success_count = 0
    for i, csv_file in enumerate(csv_files, 1):
        print(f"\n[{i}/{len(csv_files)}] 分析: {os.path.basename(csv_file)}")
        print(f"数据目录: {os.path.dirname(csv_file)}")
        
        try:
            # 创建分析器 - 会自动使用数据文件所在目录保存图表
            analyzer = RobotDataAnalyzer(csv_file)
            analyzer.generate_all_joints_analysis()
            
            success_count += 1
            print(f"✓ 完成分析，图表已保存到数据目录")
            
        except Exception as e:
            print(f"✗ 分析失败: {e}")
    
    print(f"\n批量分析完成! 成功: {success_count}/{len(csv_files)}")


# 使用示例
if __name__ == "__main__":

        folder_path = "tracking_data/20251203"
        batch_analyze_folder(folder_path)
        
        # data_file = "tracking_data/20251026/211231/uniform_motion_test_joints_speed0.4_range-1_1.csv"
        # # data_file = "tracking_data/20251025/1004/uniform_motion_test_joints_0_1_2_3_speed0.4_range-1_1_tracking_data_20251024_165905.csv"
        # analyzer = RobotDataAnalyzer(data_file)
        
        # # 为所有关节生成分析图表
        # analyzer.generate_all_joints_analysis()
        
        # # 如果只想分析特定关节，可以使用：
        # # analyzer.plot_single_joint_analysis(0)  # 分析关节0
        # # analyzer.plot_single_joint_analysis(1)  # 分析关节1
        
        # print("分析完成！")
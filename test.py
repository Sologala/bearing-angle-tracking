import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

class BezierCurve:
    def __init__(self, control_points):
        self.control_points = np.array(control_points)
        self.n = len(control_points) - 1

    def basis(self, t, i):
        return comb(self.n, i) * (t ** i) * ((1 - t) ** (self.n - i))

    def curve_point(self, t):
        point = np.zeros(3)
        for i in range(len(self.control_points)):
            point += self.basis(t, i) * self.control_points[i]
        return point

    def generate_trajectory(self, num_points):
        t = np.linspace(0, 1, num_points)
        trajectory = np.array([self.curve_point(t_i) for t_i in t])
        return trajectory

class KalmanFilter:
    def __init__(self, dim_state=3):
        self.dim_state = dim_state
        self.state = np.zeros(dim_state)
        self.P = np.eye(dim_state) * 1000  # 初始协方差
        self.Q = np.eye(dim_state) * 0.1   # 过程噪声
        self.R = 1000                       # 测量噪声

    def predict(self):
        # 简单运动模型，假设恒速运动
        # F = np.mat("1 0 0;0 1 0;0 0 0")
        # self.state = F * self.state
        self.P = self.P + self.Q

    def bearing_measurement(self, state, sensor_pos):
        # 计算方位角测量
        diff = state - sensor_pos
        norm = np.linalg.norm(diff)
        return diff / norm

    def update(self, measurement, sensor_pos):
        predicted_measurement = self.bearing_measurement(self.state, sensor_pos)
        
        # 计算雅可比矩阵
        diff = self.state - sensor_pos
        norm = np.linalg.norm(diff)
        H = np.eye(3) / norm - np.outer(diff, diff) / (norm ** 3)
        
        # Kalman增益
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T / S
        
        # 更新状态和协方差
        innovation = measurement - predicted_measurement
        self.state = self.state + K @ innovation
        self.P = (np.eye(3) - K @ H) @ self.P

def generate_measurements(trajectory, sensor_pos, noise_std=0.01):
    measurements = []
    for point in trajectory:
        # 计算方向向量
        direction = point - sensor_pos
        dist = np.linalg.norm(direction)
        direction = direction / dist
        
        dist_noised = dist + np.random.normal(0, 1, 1)[0]
        # 添加高斯噪声
        noisy_direction = direction + np.random.normal(0, noise_std, 3)
        noisy_direction = noisy_direction / np.linalg.norm(noisy_direction)
        
        measurements.append(sensor_pos + noisy_direction * dist_noised)
    return np.array(measurements)

def main():
    # 生成控制点
    control_points = [
        [-5, 5, 0],
        [5, 5, 0],
        [5, -5, 0],
        [-5, -5, 0],
        [0, 0, 4],
        [5, 5, 0]
    ]
    
    # 创建贝塞尔曲线
    bezier = BezierCurve(control_points)
    
    # 生成轨迹点
    num_points = 50
    true_trajectory = bezier.generate_trajectory(num_points)
    
    # 传感器位置
    sensor_pos = np.array([5, 5, 5])
    
    # 生成带噪声的测量
    measurements = generate_measurements(true_trajectory, sensor_pos)
    
    # 初始化Kalman滤波器
    kf = KalmanFilter()
    kf.state = true_trajectory[0] + np.random.normal(0, 0.5, 3)  # 添加初始误差
    
    # 存储估计轨迹
    estimated_trajectory = []
    
    # 运行滤波器
    for measurement in measurements:
        kf.predict()
        kf.update(measurement, sensor_pos)
        estimated_trajectory.append(kf.state.copy())
        # estimated_trajectory.append(measurement)
    
    estimated_trajectory = np.array(estimated_trajectory)
    
    # 计算轨迹误差
    trajectory_error = np.linalg.norm(estimated_trajectory - true_trajectory, axis=1)
    mean_error = np.mean(trajectory_error)
    print(f"avg error : {mean_error:.3f}")
    
    # 绘制结果
    fig = plt.figure(figsize=(12, 5))
    
    # 3D轨迹图
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(true_trajectory[:, 0], true_trajectory[:, 1], true_trajectory[:, 2], 'b-', label='true traj')
    ax1.plot(estimated_trajectory[:, 0], estimated_trajectory[:, 1], estimated_trajectory[:, 2], 'r--', label='est tra')
    ax1.scatter(sensor_pos[0], sensor_pos[1], sensor_pos[2], c='g', marker='^', s=100, label='sensor')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    ax1.set_title('3D trajs')
    
    # 误差图
    ax2 = fig.add_subplot(122)
    ax2.plot(trajectory_error)
    ax2.set_xlabel('step')
    ax2.set_ylabel('error')
    ax2.set_title('error')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

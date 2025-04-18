import open3d as o3d
import tkinter as tk
from tkinter import ttk
import threading
import numpy as np
import time
import socket
import struct
import sys

# -------------------------------
# Ring Buffer 類別：高效儲存點雲資料與時間戳
# -------------------------------
class PointRingBuffer:
    def __init__(self, max_points):
        self.max_points = max_points
        # 每筆資料包含 x, y, z 與 timestamp
        self.buffer = np.zeros((max_points, 4), dtype=np.float64)
        self.index = 0
        self.size = 0

    def add_points(self, new_points: np.ndarray):
        if new_points is None or new_points.size == 0:
            return

        if new_points.shape[1] != 4:
            raise ValueError("新加入的點必須為 Nx4 陣列（包含 x, y, z, timestamp）")

        n = new_points.shape[0]
        end_index = self.index + n

        if end_index <= self.max_points:
            self.buffer[self.index:end_index] = new_points
        else:
            overflow = end_index - self.max_points
            self.buffer[self.index:] = new_points[:n - overflow]
            self.buffer[:overflow] = new_points[n - overflow:]

        self.index = (self.index + n) % self.max_points
        self.size = min(self.size + n, self.max_points)

    def get_recent_points(self, retention_time):
        if self.size == 0:
            return np.zeros((0, 3), dtype=np.float32)

        size = int(self.size)
        now = time.time()
        valid = (now - self.buffer[:size, 3]) <= retention_time
        return self.buffer[:size][valid][:, :3].astype(np.float32)
    
    def clear(self):
        self.buffer[:] = 0
        self.index = 0
        self.size = 0

# -------------------------------
# UDP 接收器（使用 RingBuffer 儲存）
# -------------------------------
class UDPReceiver(threading.Thread):
    def __init__(self, host, port, ring_buffer, lock):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.ring_buffer = ring_buffer
        self.lock = lock

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.host, self.port))
        print(f"UDP 伺服器正在監聽 {self.host}:{self.port}")
        while True:
            data, addr = sock.recvfrom(65535)
            if not data or len(data) < 4:
                continue

            # 前 4 個 byte 代表點的數量
            num_points = struct.unpack('<I', data[:4])[0]
            expected_size = 4 + num_points * 3 * 4
            if len(data) != expected_size:
                continue

            floats = struct.unpack('<' + 'f' * (num_points * 3), data[4:])
            pts = np.array(floats, dtype=np.float64).reshape(-1, 3)

            # 每個點加入系統時間戳記
            t_now = time.time()
            timestamps = np.full((pts.shape[0], 1), t_now, dtype=np.float64)
            new_points = np.hstack((pts, timestamps))  # shape: [x, y, z, timestamp]

            with self.lock:
                self.ring_buffer.add_points(new_points)

# -------------------------------
# 使用 Open3D 與 tkinter 製作點雲視覺化與控制面板
# -------------------------------
class PointCloudViewer:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        
        # 參數設定：點雲顯示時長與顏色漸層最大距離
        self.retention_seconds = 5.0
        self.max_distance = 5.0

        self.ring_buffer = PointRingBuffer(max_points=1920000)
        self.points_lock = threading.Lock()
        
        # 啟動 UDP 接收 (監聽 8080 埠)
        self.udp_receiver = UDPReceiver('0.0.0.0', 8080, self.ring_buffer, self.points_lock)
        self.udp_receiver.start()
        
        # 啟動 Open3D 視窗更新執行緒（用於點雲顯示）
        self.o3d_thread = threading.Thread(target=self.update_open3d_visualizer, daemon=True)
        self.o3d_thread.start()

        # 建立 tkinter 控制面板 (必須在主執行緒中運行)
        self.setup_tkinter()

    def update_open3d_visualizer(self):
        # 建立 Open3D 視窗與初始點雲 geometry
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="UDP Point Cloud Viewer (Open3D)", width=self.width, height=self.height)
        pcd = o3d.geometry.PointCloud()
        vis.add_geometry(pcd)

        # 加入座標軸參考 (可選)
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        vis.add_geometry(coordinate_frame)

        while True:
            with self.points_lock:
                pts = self.ring_buffer.get_recent_points(self.retention_seconds)
            if pts.shape[0] > 0:
                # 更新點雲幾何體
                pcd.points = o3d.utility.Vector3dVector(pts)
                # 計算每個點與原點的距離
                distances = np.linalg.norm(pts, axis=1)
                # 計算顏色插值因子（0 到 1），依據 max_distance
                d = np.clip(distances / self.max_distance, 0.0, 1.0).reshape(-1, 1)
                # 定義近點顏色 (黃色) 與遠點顏色 (藍色)
                near_color = np.array([1.0, 1.0, 0.0])
                far_color = np.array([0.0, 0.5, 1.0])
                colors = (1 - d) * near_color + d * far_color
                pcd.colors = o3d.utility.Vector3dVector(colors)
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.1)

    def save_points_to_ply(self):
        with self.points_lock:
            pts = self.ring_buffer.get_recent_points(self.retention_seconds)
        if pts.size == 0:
            print("沒有可儲存的點雲資料")
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"pointcloud_{timestamp}.ply"
        with open(filename, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(pts)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            for p in pts:
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
        print(f"已儲存 {len(pts)} 筆點雲資料至 {filename}")

    def clear_point_cloud(self):
        with self.points_lock:
            self.ring_buffer.clear()

    def setup_tkinter(self):
        self.root = tk.Tk()
        self.root.title("控制面板")
        self.root.geometry("300x220")

        # 保留時間控制（Storage seconds）
        retention_frame = ttk.Frame(self.root)
        retention_frame.pack(pady=5)
        ttk.Label(retention_frame, text="Storage seconds").pack(side=tk.LEFT)
        self.retention_var = tk.DoubleVar(value=self.retention_seconds)
        retention_scale = tk.Scale(retention_frame, variable=self.retention_var,
                                   from_=1.0, to=30.0, resolution=0.5,
                                   orient=tk.HORIZONTAL, command=self.on_retention_change)
        retention_scale.pack(side=tk.LEFT)

        # 顏色最大距離控制 (Max distance)
        max_distance_frame = ttk.Frame(self.root)
        max_distance_frame.pack(pady=5)
        ttk.Label(max_distance_frame, text="Max distance").pack(side=tk.LEFT)
        self.max_distance_var = tk.DoubleVar(value=self.max_distance)
        max_distance_scale = tk.Scale(max_distance_frame, variable=self.max_distance_var,
                                      from_=1.0, to=20.0, resolution=0.5,
                                      orient=tk.HORIZONTAL, command=self.on_max_distance_change)
        max_distance_scale.pack(side=tk.LEFT)

        # 清除點雲按鈕
        clear_button = ttk.Button(self.root, text="Clear Point Cloud", command=self.clear_point_cloud)
        clear_button.pack(pady=5)
        # 儲存 PLY 按鈕
        save_button = ttk.Button(self.root, text="Save to .PLY", command=self.save_points_to_ply)
        save_button.pack(pady=5)

        self.status_label = ttk.Label(self.root, text="正在更新點雲資料...")
        self.status_label.pack(pady=5)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def on_retention_change(self, val):
        self.retention_seconds = float(val)
        self.status_label.config(text=f"Storage seconds: {self.retention_seconds:.1f}")

    def on_max_distance_change(self, val):
        self.max_distance = float(val)
        self.status_label.config(text=f"Max distance: {self.max_distance:.1f}")

    def on_closing(self):
        # 關閉 tkinter 後結束整個程式
        self.root.destroy()
        sys.exit(0)

if __name__ == '__main__':
    viewer = PointCloudViewer()

import cv2
import cv2.aruco as aruco
import numpy as np
import threading
import time
import glfw
from OpenGL.GL import *
import pyrr
import socket
import struct
import imgui
import json
import os  # 🔧 在檔案頂部加入
from imgui.integrations.glfw import GlfwRenderer
from scipy.spatial import cKDTree  # 此處未直接使用，但可供擴充
import tkinter as tk
from tkinter import filedialog, messagebox
import torch
# =============================================================================
# ★★  Multi‑ArUco 量測表  ★★
#   build_transform(tx,ty,tz, yaw,pitch,roll)  角度單位 deg, 先旋轉後平移
# =============================================================================
def build_transform(tx, ty, tz, yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0):
    rx, ry, rz = np.radians([roll_deg, pitch_deg, yaw_deg])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rz = np.array([[cz, -sz, 0, 0],[sz, cz, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]], np.float32)
    Ry = np.array([[cy, 0, sy, 0],[0, 1, 0, 0],[-sy, 0, cy, 0],[0, 0, 0, 1]], np.float32)
    Rx = np.array([[1, 0, 0, 0],[0, cx, -sx, 0],[0, sx, cx, 0],[0, 0, 0, 1]], np.float32)
    T  = np.eye(4, dtype=np.float32);  T[:3, 3] = [tx, ty, tz]
    return T @ Rz @ Ry @ Rx

GLOBAL_TO_MARKER = {
    0: np.eye(4, dtype=np.float32),              # id 0 為世界原點
    1: build_transform(0.00, 1.00, 0.00),     # 以下請依手動量測填值
    2: build_transform(0.00, 1.00, 0.000),
    3: build_transform(0.00, 2.00, 0.000),
    4: build_transform(0.00, 3.00, 0.000),
    5: build_transform(0.00, 4.00, 0.000),
    6: build_transform(0.00, 5.00, 0.000),
    7: build_transform(0.00, 6.00, 0.000),
    8: build_transform(0.00, 8.00, 0.000),
    9: build_transform(0.00, 10.00, 0.000),
    10: build_transform(0.00, 12.00, 0.000),
    # …持續新增
}

# =============================================================================
# ★★  全域點雲緩衝 (按鈕觸發才 append)  ★★
# =============================================================================
GLOBAL_MAX_POINTS = 20_000_000
global_pts  = np.zeros((GLOBAL_MAX_POINTS, 3), np.float32)
global_size = 0
global_lock = threading.Lock()
# 全域座標系緩存：key → 4×4 矩陣
global_coords = {}


def add_global_coord(name: str, mat: np.ndarray):
    """把 name:4×4 矩陣 加到全域座標系，重複 name 會直接覆蓋"""
    global_coords[name] = mat.copy()


def add_to_global(new_pts: np.ndarray):
    """把 new_pts(N,3) 追加到 global_pts；滿就覆蓋最舊"""
    global global_size
    n = new_pts.shape[0]
    if n == 0: return
    with global_lock:
        if global_size + n > GLOBAL_MAX_POINTS:
            keep = GLOBAL_MAX_POINTS - n
            if keep > 0:
                global_pts[:keep] = global_pts[global_size - keep:global_size]
            global_size = keep
        global_pts[global_size:global_size + n] = new_pts
        global_size += n

# =============================================================================
# 全域變數：ArUco 轉換矩陣（4x4）與鎖
# =============================================================================
global_transform_lock = threading.Lock()
global_transform = np.eye(4, dtype=np.float32)  # 初始值為單位矩陣
latest_transform_lock = threading.Lock()
latest_transform_matrix = np.eye(4, dtype=np.float32)

# =============================================================================
# 相機校正參數（請根據實際校正結果修改）
# =============================================================================
camera_matrix = np.array([[2334.26147, 0, 1003.37764],
                          [0, 2339.77674, 510.779799],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.array([ 0.09009018, 0.1080235,  -0.01002997,  0.00863418, -1.06185542], dtype=np.float32)

# =============================================================================
# ArUco 偵測執行緒：持續偵測並更新 4x4 變換矩陣
# =============================================================================
def detect_aruco_thread(camera_id=0, dictionary=aruco.DICT_4X4_250):
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_ZOOM, 200)
    cap.set(cv2.CAP_PROP_EXPOSURE,-6)
    cap.set(cv2.CAP_PROP_GAIN, 3)
    cap.set(cv2.CAP_PROP_SETTINGS, 1)
    if not cap.isOpened():
        print("無法打開攝像頭")
        return

    aruco_dict = aruco.getPredefinedDictionary(dictionary)
    parameters = aruco.DetectorParameters()

    global global_transform
    while True:
        ret, frame = cap.read()
        if not ret:
            print("無法從攝像頭取得影像")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        # ---------- 同時處理多顆 ArUco ----------
        if ids is not None:
            frame = aruco.drawDetectedMarkers(frame, corners, ids)
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, 0.19, camera_matrix, dist_coeffs)

            for i, id_arr in enumerate(ids):
                mid = int(id_arr[0])
                if mid not in GLOBAL_TO_MARKER:        # 未登錄則跳過
                    continue
                rotM, _ = cv2.Rodrigues(rvecs[i][0])
                Cam_T_marker = np.eye(4, dtype=np.float32)
                Cam_T_marker[:3, :3] = rotM
                Cam_T_marker[:3, 3]  = tvecs[i][0]

                Global_T_marker = GLOBAL_TO_MARKER[mid]   # 原點→marker
                Cam_T_Global = Cam_T_marker @ np.linalg.inv(Global_T_marker)

                with latest_transform_lock:
                    latest_transform_matrix[:] = Cam_T_Global
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs,
                                  rvecs[i][0], tvecs[i][0], 0.1)
                break   # 找到一顆即可

        cv2.imshow('ArUco Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# =============================================================================
# 點雲轉換函數：利用 4x4 齊次變換矩陣轉換 (N, 3) 點雲資料
# =============================================================================
def transform_point_cloud_(points, Lidar_T_Aruco):
    """
    參數:
      points (np.ndarray): (N, 3) 點雲資料
      transform_matrix (np.ndarray): 4x4 齊次變換矩陣
    回傳:
      轉換後的 (N, 3) 點雲資料
    """
    if points.size == 0:
        return points
    ones = np.ones((points.shape[0], 1), dtype=points.dtype)
    points_hom = np.hstack((points, ones))
    arTlidar = np.linalg.inv(Lidar_T_Aruco) #  # ArUco 到 LiDAR 的變換矩陣
    transformed = (arTlidar @ points_hom.T).T
    
    return transformed[:, :3]

def transform_point_cloud(points, Lidar_T_Aruco, device='cuda'):
    """
    參數:
      points (np.ndarray): (N, 3) 點雲資料
      transform_matrix (np.ndarray): 4x4 齊次變換矩陣
    回傳:
      轉換後的 (N, 3) 點雲資料
    """
    if points.size == 0:
        return points
    pts = torch.tensor(points, dtype=torch.float32, device=device)
    ones = torch.ones((pts.shape[0], 1), dtype=torch.float32, device=device)
    pts_hom = torch.cat([pts, ones], dim=1)  # (N,4)
    mat = torch.tensor(np.linalg.inv(Lidar_T_Aruco), dtype=torch.float32, device=device)
    transformed = pts_hom @ mat.T  # (N,4) @ (4,4)
    return transformed[:, :3].cpu().numpy()  # 回 numpy 給 OpenGL/其他後續處理


# =============================================================================
# 相機到光達外參：定義一個 4x4 齊次矩陣（示例數值，可根據實際校正修改）
# =============================================================================
dx, dy, dz = 0.0, 0.075, -0.023  # 平移量
theta = np.radians(-45-22.5)  # Z 軸旋轉角度（逆時針）

cos_t, sin_t = np.cos(theta), np.sin(theta)

# Z 軸旋轉矩陣（齊次）
Rz = np.array([
    [cos_t, -sin_t, 0, 0],
    [sin_t,  cos_t, 0, 0],
    [0,      0,     1, 0],
    [0,      0,     0, 1]
], dtype=np.float32)


theta_x = np.radians(-25.0)                # 30°
cos_x, sin_x = np.cos(theta_x), np.sin(theta_x)

Rx = np.array([
    [1,     0,      0, 0],
    [0, cos_x, -sin_x, 0],
    [0, sin_x,  cos_x, 0],
    [0,     0,      0, 1]
], dtype=np.float32)

# 3. 組合旋轉矩陣
#    ⬇️ 依「先 X、再 Z」的次序
#    若希望先 Z 再 X，請改成 Rz @ Rx
# ──────────────────────────────
R = Rx @ Rz          # 先對 X 轉 30°，再對 Z 轉 -67.5°

# 平移矩陣
T = np.eye(4, dtype=np.float32)
T[:3, 3] = [dx, dy, dz]

# Camera to LiDAR 的外參（先旋轉再平移）
Cam_T_Lidar = T @ R

# =============================================================================
# PointRingBuffer 類別：環狀點雲資料緩衝區
# =============================================================================
class PointRingBuffer:
    def __init__(self, max_points):
        self.max_points = max_points
        self.buffer = np.zeros((max_points, 4), dtype=np.float32)  # 存 (x,y,z,timestamp)
        self.index = 0
        self.size = 0

    def add_points(self, new_points: np.ndarray):
        if new_points is None or new_points.size == 0:
            return
        if new_points.shape[1] != 4:
            raise ValueError("新加入的點必須為 Nx4 陣列（x,y,z,timestamp）")
        n = new_points.shape[0]
        end_index = self.index + n
        if end_index <= self.max_points:
            self.buffer[self.index:end_index] = new_points
        else:
            overflow = end_index - self.max_points
            self.buffer[self.index:] = new_points[:n-overflow]
            self.buffer[:overflow] = new_points[n-overflow:]
        self.index = (self.index + n) % self.max_points
        self.size = min(self.size + n, self.max_points)

    # def get_recent_points(self, retention_time, with_time=False):
    #     if self.size == 0:
    #         if with_time:
    #             return np.empty((0, 4), dtype=np.float32)
    #         else:
    #             return np.empty((0, 3), dtype=np.float32)

    #     now = time.monotonic()
    #     cutoff = now - retention_time

    #     if self.size < self.max_points:
    #         data = self.buffer[:self.size]
    #     else:
    #         data = np.vstack((self.buffer[self.index:], self.buffer[:self.index]))

    #     idx = np.searchsorted(data[:, 3], cutoff, side='left')
    #     valid = data[idx:]

    #     if with_time:
    #         return valid  # (N, 4)
    #     else:
    #         return valid[:, :3]  # (N, 3)
        
    def get_recent_points(self, retention_time, with_time=False):
        if self.size == 0:
            return np.empty((0, 4 if with_time else 3), dtype=np.float32)
        now = time.monotonic()
        cutoff = now - retention_time

        if self.size < self.max_points:
            data = self.buffer[:self.size]
            idx = np.searchsorted(data[:, 3], cutoff, side='left')
            valid = data[idx:]
            return valid if with_time else valid[:, :3]
        else:
            buf1 = self.buffer[self.index:]
            buf2 = self.buffer[:self.index]
            idx1 = np.searchsorted(buf1[:, 3], cutoff, side='left')
            valid1 = buf1[idx1:] if idx1 < len(buf1) else np.empty((0, 4), dtype=np.float32)
            valid2 = buf2[buf2[:, 3] >= cutoff] if buf2.size > 0 else np.empty((0, 4), dtype=np.float32)
            if with_time:
                return np.concatenate((valid1, valid2), axis=0)
            else:
                return np.concatenate((valid1[:, :3], valid2[:, :3]), axis=0)
   


    def clear(self):
        self.buffer[:] = 0
        self.index = 0
        self.size = 0

# =============================================================================
# UDPReceiver 類別：接收 UDP 封包、解碼後將點雲資料存入環狀緩衝區
# =============================================================================
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
            num_points = struct.unpack('<I', data[:4])[0]
            expected_size = 4 + num_points * 3 * 4
            if len(data) != expected_size:
                continue
            floats = struct.unpack('<' + 'f' * (num_points * 3), data[4:])
            pts = np.array(floats, dtype=np.float32).reshape(-1, 3)
            t_now = time.monotonic()
            timestamps = np.full((pts.shape[0], 1), t_now, dtype=np.float32)
            new_points = np.hstack((pts, timestamps))
            with self.lock:
                self.ring_buffer.add_points(new_points)


# 🔧 自訂 JSON 編碼器：讓 numpy 的矩陣與數值可以轉成 JSON
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # 把 numpy 矩陣轉成 list
        elif isinstance(obj, (np.float32, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int32)):
            return int(obj)
        return super().default(obj)

# =============================================================================
# Shader 工具函式：編譯與連結 shader 程式
# =============================================================================
def compile_shader(shader_type, source):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
        error = glGetShaderInfoLog(shader).decode()
        raise RuntimeError("Shader compile error: " + error)
    return shader

def create_shader_program(vertex_src, fragment_src):
    vs = compile_shader(GL_VERTEX_SHADER, vertex_src)
    fs = compile_shader(GL_FRAGMENT_SHADER, fragment_src)
    program = glCreateProgram()
    glAttachShader(program, vs)
    glAttachShader(program, fs)
    glLinkProgram(program)
    if glGetProgramiv(program, GL_LINK_STATUS) != GL_TRUE:
        error = glGetProgramInfoLog(program).decode()
        raise RuntimeError("Shader link error: " + error)
    glDeleteShader(vs)
    glDeleteShader(fs)
    return program

# =============================================================================
# Vertex Shader：將點雲從物件空間轉換至 clip space，同時計算到原點的距離
# =============================================================================
vertex_shader_source = """
#version 330 core
layout(location = 0) in vec3 aPos;
uniform mat4 MVP;
out float vDistance;
void main(){
    vDistance = length(aPos);
    gl_Position = MVP * vec4(aPos, 1.0);
}
"""

# =============================================================================
# Fragment Shader：使用 Jet Colormap 著色或統一顏色
# =============================================================================
fragment_shader_source = """
#version 330 core
in float vDistance;
out vec4 FragColor;
uniform bool useUniformColor;
uniform vec4 uColor;
uniform float maxDistance;
vec3 jetColor(float t) {
    float r = clamp(min(4.0 * t - 1.5, -4.0 * t + 4.5), 0.0, 1.0);
    float g = clamp(min(4.0 * t - 0.5, -4.0 * t + 3.5), 0.0, 1.0);
    float b = clamp(min(4.0 * t + 0.5, -4.0 * t + 2.5), 0.0, 1.0);
    return vec3(r, g, b);
}
void main(){
    if(useUniformColor)
        FragColor = uColor;
    else {
        float d = clamp(vDistance / maxDistance, 0.0, 1.0);
        FragColor = vec4(jetColor(d), 1.0);
    }
}
"""

# =============================================================================
# PointCloudViewer 類別：負責點雲視覺化與使用者互動
# =============================================================================
class PointCloudViewer:
    def __init__(self, width=1600, height=800, enable_network=True):
        self.width = width
        self.height = height
        self.rotation_x = 0.0  
        self.rotation_y = 0.0  
        self.last_cursor_pos = None
        self.zoom = 20.0
        self.retention_seconds = 5.0
        self.pan_offset = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.ring_buffer = PointRingBuffer(max_points=10000000)
        self.points_lock = threading.Lock()
        # 0 = Live, 1 = Global, 2 = File
        self.send_mode = 0
        self.save_source = 0
        
        self.last_live_fetch_time = 0.0
        self.live_pts_cache = (np.empty((0, 3), np.float32), np.empty((0, 1), np.float32))

        # 啟動 UDP 接收線程
        if enable_network:
            self.udp_receiver = UDPReceiver('0.0.0.0', 8080, self.ring_buffer, self.points_lock)
            self.udp_receiver.start()

        # 初始化 GLFW、ImGui、Shader 與 OpenGL 緩衝區
        self.init_glfw()
        self.init_imgui()
        self.init_shaders()
        self.init_buffers()
        self._axis_vao_cache = {}  # cache for reused axis

        # 啟動 TCP 傳送（此處示例連線至指定的 IP 與埠）
        # 初始化可調整的 TCP/IP 參數
        self.tcp_host = "192.168.137.1"
        #self.tcp_host = "127.0.0.1"
        self.tcp_port = 9000
        # 啟動 TCP 傳送（使用變數 self.tcp_host, self.tcp_port）
        if enable_network:
            self._tcp_stop_event = threading.Event()
            self.start_tcp_thread()
        # 定義預設的坐標系資訊，以 identity matrix 作為範例
        self.coordinate_system = {
            "matrix": list(pyrr.matrix44.create_identity(dtype=np.float32))
        }

        self.Word_Point = None
        self.Camera_Position = None 
        self.Lidar_Position =  None
        self.Lidar_T_Aruco = np.eye(4, dtype=np.float32)

        self.use_live_data = True          # True ➜ 即時模式；False ➜ 載入檔案模式
        self.loaded_points = np.empty((0, 3), dtype=np.float32)
        self.loaded_poses  = {}            # 儲存 camera/lidar/world 4×4
        self.loaded_points = np.empty((0, 4), dtype=np.float32)  # x,y,z,timestamp
        self.loaded_poses  = {}

        # 新增：檔案模式時間範圍與滑桿參數
        self.file_time_min   = 0.0
        self.file_time_max   = 0.0
        self.file_time_start = 0.0
        self.file_time_end   = 0.0

        self.show_global = False            # 是否顯示 global 點雲
        self.global_color = (1.0, 1.0, 1.0, 1.0)  # 畫 global 點的單一顏色

        # 在 __init__ 中加入（設定預設為 ±5m 的 3D 空間）
        self.fence_min = np.array([-1.0, -1.0, 0.0], dtype=np.float32)
        self.fence_max = np.array([ 1.0,  1.0,  1.7], dtype=np.float32)
        self.use_inner_fence  = True  # 控制是否啟用電子圍籬
        
        self.use_outer_fence = False  # 新增的 Outer Fence
        self.fence_outer_min = np.array([-3.0, -3.0, -1.0], dtype=np.float32)
        self.fence_outer_max = np.array([ 3.0,  3.0,  3.0], dtype=np.float32)
        
        self.enable_voxel = False        # 是否啟用下採樣
        self.voxel_size = 0.03           # 預設體素格邊長



    def start_tcp_thread(self):
        self._tcp_stop_event.clear()
        self.tcp_sender_thread = threading.Thread(
            target=self.tcp_sender,
            daemon=True
        )
        self.tcp_sender_thread.start()


    def restart_tcp_connection(self):
        # 使用時從外部呼叫
        print("TCP 傳送線程：正在重啟…")
        # 先通知舊線程停止
        self._tcp_stop_event.set()
        # 等待舊線程結束
        if self.tcp_sender_thread.is_alive():
            self.tcp_sender_thread.join(timeout=3.0)
        # 再啟動新線程
        self.start_tcp_thread()
    

    def tcp_sender(self):
        import socket, time, struct, json
        import numpy as np
        global global_coords

        stop_event = self._tcp_stop_event
        host, port = self.tcp_host, self.tcp_port

        while not stop_event.is_set():
            # —— 嘗試建立連線 —— #
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5.0)  # 連線／send 最多等 5 秒
                sock.connect((host, port))
                sock.settimeout(None)
                print(f"TCP Sender: 已連線至 {host}:{port}")
            except Exception as e:
                print(f"TCP Sender 連線失敗: {e}，2 秒後重試…")
                time.sleep(2)
                continue

            # —— 連線成功後進入傳送迴圈 —— #
            try:
                while not stop_event.is_set():
                    # 準備 coord 系列
                    coord_system = {
                        name: mat.tolist()
                        for name, mat in global_coords.items()
                        if name.startswith("marker_")
                    }
                    # ► 準備要送的點雲資料（Live / Global / File）
                    if self.send_mode == 0:
                        with self.points_lock:
                            pts = self.ring_buffer.get_recent_points(self.retention_seconds)
                        pts3, stamps = self.get_latest_live_points()  # stamps 是 (N,1)
                        data4 = np.hstack((pts3, stamps))
                        coord_system = {
                            "lidar":  self.Lidar_Position,
                            "camera": self.Camera_Position,
                            "world":  self.Word_Point,
                        }
                    elif self.send_mode == 1:
                        with global_lock:
                            glob = global_pts[:global_size].copy()
                        stamps = np.full((glob.shape[0],1), time.monotonic(), dtype=np.float32)
                        data4 = np.hstack((glob, stamps))
                        coord_system = {
                            name: mat
                            for name, mat in global_coords.items()
                        }
                    else:  # File 模式
                        mask = np.logical_and(
                            self.loaded_points[:,3] >= self.file_time_start,
                            self.loaded_points[:,3] <= self.file_time_end
                        )
                        pts = self.loaded_points[mask, :3].astype(np.float32)
                        stamps = np.full((pts.shape[0],1), time.monotonic(), dtype=np.float32)
                        data4 = np.hstack((pts, stamps))
                        coord_system = self.loaded_poses.copy()

                    # 拆出 XYZ + timestamp bytes
                    points_bin = data4.astype(np.float32).tobytes()
                    coord_bin = json.dumps(coord_system, cls=NumpyEncoder).encode('utf-8')

                    # Header：coord 長度 + points 長度
                    header = struct.pack('<II', len(coord_bin), len(points_bin))
                    message = header + coord_bin + points_bin

                    sock.sendall(message)
                    time.sleep(1)

            except Exception as e:
                print(f"TCP Sender 傳送失敗: {e}，2 秒後重試…")
                time.sleep(2)

            finally:
                sock.close()

        print("TCP Sender 線程已停止")

    def get_latest_live_points(self):
        """取得經 ArUco 轉換後的即時點雲資料（以 LiDAR 當中心電子圍籬）"""
        now = time.time()
        if now - self.last_live_fetch_time < 0.2:  # 每 0.2秒更新一次
            return self.live_pts_cache

        with self.points_lock:
            pts_full = self.ring_buffer.get_recent_points(self.retention_seconds, with_time=True)

        if pts_full.size == 0:
            self.live_pts_cache = (np.empty((0, 3), dtype=np.float32), np.empty((0, 1), dtype=np.float32))
            self.last_live_fetch_time = now
            return self.live_pts_cache

        pts_xyz = pts_full[:, :3]
        pts_time = pts_full[:, 3:4]

        with global_transform_lock:
            Camera_T_Aruco = global_transform.copy()

        # 🔥 先更新 Camera_Position 和 Lidar_Position
        self.Camera_Position = np.linalg.inv(Camera_T_Aruco)
        self.Lidar_Position  = self.Camera_Position @ Cam_T_Lidar

        # 然後建立 LiDAR 到 ArUco 世界座標的變換
        self.Lidar_T_Aruco = np.linalg.inv(Cam_T_Lidar) @ Camera_T_Aruco

        # 然後轉換點雲
        pts3 = transform_point_cloud(pts_xyz, self.Lidar_T_Aruco)

        # 🔥 轉成 LiDAR 局部座標
        if self.Lidar_Position is None:
            lidar_inv = np.eye(4, dtype=np.float32)
        else:
            lidar_inv = np.linalg.inv(self.Lidar_Position)

        ones = np.ones((pts3.shape[0], 1), dtype=np.float32)
        pts3_hom = np.hstack((pts3, ones))  # (N,4)
        pts3_local = (lidar_inv @ pts3_hom.T).T[:, :3]  # (N,3)，轉到LiDAR自身座標系

        # # ✅ 電子圍籬篩選
        # if self.use_fence and pts3_local.shape[0] > 0:
        #     #mask = np.all((pts3_local >= self.fence_min) & (pts3_local <= self.fence_max), axis=1)
        #     mask = ~np.all((pts3_local >= self.fence_min) & (pts3_local <= self.fence_max), axis=1)
        #     pts3 = pts3[mask]
        #     pts_time = pts_time[mask]

        if pts3_local.shape[0] > 0:
            mask = np.ones((pts3_local.shape[0],), dtype=bool)
            if self.use_inner_fence:
                inner_mask = np.all((pts3_local >= self.fence_min) & (pts3_local <= self.fence_max), axis=1)
                mask &= ~inner_mask  # 把範圍內的剔除
            if self.use_outer_fence:
                outer_mask = np.all((pts3_local >= self.fence_outer_min) & (pts3_local <= self.fence_outer_max), axis=1)
                mask &= outer_mask  # 只保留範圍內的
            pts3 = pts3[mask]
            pts_time = pts_time[mask]


        # 更新 cache

        self.live_pts_cache = (pts3, pts_time)
        self.last_live_fetch_time = now
        return self.live_pts_cache



    def init_glfw(self):
        if not glfw.init():
            raise Exception("GLFW 初始化失敗")
        self.window = glfw.create_window(self.width, self.height, "UDP Point Cloud Viewer", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("無法建立視窗")
        glfw.make_context_current(self.window)
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_cursor_pos_callback(self.window, self.cursor_pos_callback)
        glfw.set_framebuffer_size_callback(self.window, self.framebuffer_size_callback)

        glEnable(GL_DEPTH_TEST)

    def init_imgui(self):
        # 每个 Viewer 都建立自己独立的 ImGui context
        self.imgui_ctx = imgui.create_context()
        imgui.style_colors_dark()           # 喜欢别的风格随意改
        # 绑定到当前窗口
        imgui.set_current_context(self.imgui_ctx)
        self.impl = GlfwRenderer(self.window)



    def init_shaders(self):
        self.shader_program = create_shader_program(vertex_shader_source, fragment_shader_source)
        self.mvp_loc = glGetUniformLocation(self.shader_program, "MVP")
        self.max_distance_loc = glGetUniformLocation(self.shader_program, "maxDistance")
        self.max_distance = 5.0
    
    def framebuffer_size_callback(self, window, width, height):
        self.width = width
        self.height = height
        glViewport(0, 0, width, height)
        safe_height = height if height > 0 else 1  # 防止除以零
        self.projection = pyrr.matrix44.create_perspective_projection_matrix(
            45, width / safe_height, 0.1, 100.0
        )



    def init_buffers(self):
        self.max_points = 19200000
        self.point_vao = glGenVertexArrays(1)
        self.point_vbo = glGenBuffers(1)
        glBindVertexArray(self.point_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.point_vbo)
        glBufferData(GL_ARRAY_BUFFER, self.max_points * 3 * 4, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glBindVertexArray(0)

        # 建立背景格線
        grid_size = 10
        grid_lines = []
        for i in range(-grid_size, grid_size + 1):
            grid_lines.extend([-grid_size, i, 0.0, grid_size, i, 0.0])
            grid_lines.extend([i, -grid_size, 0.0, i, grid_size, 0.0])
        self.grid_vertices = np.array(grid_lines, dtype=np.float32)
        self.grid_vertex_count = len(grid_lines) // 3
        self.grid_vao = glGenVertexArrays(1)
        self.grid_vbo = glGenBuffers(1)
        glBindVertexArray(self.grid_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.grid_vbo)
        glBufferData(GL_ARRAY_BUFFER, self.grid_vertices.nbytes, self.grid_vertices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glBindVertexArray(0)

        # 建立 XYZ 軸線
        axes_vertices = np.array([
            0, 0, 0, 1, 0, 0,  # X 軸（紅色）
            0, 0, 0, 0, 1, 0,  # Y 軸（綠色）
            0, 0, 0, 0, 0, 1   # Z 軸（藍色）
        ], dtype=np.float32)
        self.axes_vao = glGenVertexArrays(1)
        self.axes_vbo = glGenBuffers(1)
        glBindVertexArray(self.axes_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.axes_vbo)
        glBufferData(GL_ARRAY_BUFFER, axes_vertices.nbytes, axes_vertices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glBindVertexArray(0)

        self.projection = pyrr.matrix44.create_perspective_projection_matrix(
            45, self.width / self.height, 0.1, 100.0
        )

    def mouse_button_callback(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                self.last_cursor_pos = glfw.get_cursor_pos(window)
            elif action == glfw.RELEASE:
                self.last_cursor_pos = None

    def cursor_pos_callback(self, window, xpos, ypos):
        self.last_cursor_pos = (xpos, ypos)

    def handle_mouse_input(self):
        if self.last_cursor_pos:
            xpos, ypos = glfw.get_cursor_pos(self.window)
            last_x, last_y = self.last_cursor_pos
            sensitivity = 0.005
            dx = xpos - last_x
            dy = ypos - last_y
            self.rotation_y += dx * sensitivity
            self.rotation_x += dy * sensitivity
            max_pitch = np.pi / 2 - 0.01
            self.rotation_x = np.clip(self.rotation_x, -max_pitch, max_pitch)
            self.last_cursor_pos = (xpos, ypos)

    def handle_keyboard_input(self):
        rotation_speed = 0.01
        pan_step = 0.1

        if glfw.get_key(self.window, glfw.KEY_LEFT) == glfw.PRESS:
            self.rotation_y -= rotation_speed
        if glfw.get_key(self.window, glfw.KEY_RIGHT) == glfw.PRESS:
            self.rotation_y += rotation_speed
        if glfw.get_key(self.window, glfw.KEY_UP) == glfw.PRESS:
            self.rotation_x -= rotation_speed
        if glfw.get_key(self.window, glfw.KEY_DOWN) == glfw.PRESS:
            self.rotation_x += rotation_speed

        yaw_matrix = pyrr.matrix33.create_from_z_rotation(self.rotation_y)
        pitch_matrix = pyrr.matrix33.create_from_x_rotation(self.rotation_x)
        rotation_matrix = pyrr.matrix33.multiply(pitch_matrix, yaw_matrix)

        forward = pyrr.vector3.normalize(rotation_matrix @ np.array([0, 0, -1], dtype=np.float32))
        right   = pyrr.vector3.normalize(rotation_matrix @ np.array([1, 0, 0], dtype=np.float32))
        up      = pyrr.vector3.normalize(np.array([0, 1, 0], dtype=np.float32))

        if glfw.get_key(self.window, glfw.KEY_W) == glfw.PRESS:
            self.pan_offset += up * pan_step
        if glfw.get_key(self.window, glfw.KEY_S) == glfw.PRESS:
            self.pan_offset -= up * pan_step
        if glfw.get_key(self.window, glfw.KEY_A) == glfw.PRESS:
            self.pan_offset -= right * pan_step
        if glfw.get_key(self.window, glfw.KEY_D) == glfw.PRESS:
            self.pan_offset += right * pan_step
        if glfw.get_key(self.window, glfw.KEY_Q) == glfw.PRESS:
            self.pan_offset += forward * pan_step
        if glfw.get_key(self.window, glfw.KEY_E) == glfw.PRESS:
            self.pan_offset -= forward * pan_step

        zoom_step = 0.1
        if glfw.get_key(self.window, glfw.KEY_PAGE_UP) == glfw.PRESS:
            self.zoom = max(0.1, self.zoom - zoom_step)
        if glfw.get_key(self.window, glfw.KEY_PAGE_DOWN) == glfw.PRESS:
            self.zoom += zoom_step

    def handle_scroll_input(self):
        io = imgui.get_io()
        if io.mouse_wheel != 0.0:
            self.zoom = max(self.zoom - io.mouse_wheel * 1.0, 0.1)
            io.mouse_wheel = 0.0

    def update(self):
        self.handle_keyboard_input()
        self.handle_mouse_input()
        self.handle_scroll_input()

    def render(self):
        global global_size
        # 清除畫面
        glClearColor(0.2, 0.2, 0.2, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # -------------------------
        # 建立 view 與模型矩陣
        # -------------------------
        # 建立 view 矩陣
        base_eye = np.array([0.0, -self.zoom, 5.0], dtype=np.float32)
        base_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        eye = base_eye + self.pan_offset
        target = base_target + self.pan_offset
        view = pyrr.matrix44.create_look_at(eye, target, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        # 建立模型矩陣（旋轉）
        pitch = pyrr.matrix44.create_from_x_rotation(self.rotation_x)
        yaw   = pyrr.matrix44.create_from_z_rotation(self.rotation_y) 
        model = pyrr.matrix44.multiply(yaw, pitch)
        # 計算 MVP 矩陣
        MVP = pyrr.matrix44.multiply(model, view)
        MVP = pyrr.matrix44.multiply(MVP, self.projection)

        # 傳入 shader
        glUseProgram(self.shader_program)
        glUniformMatrix4fv(self.mvp_loc, 1, GL_FALSE, MVP)
        glUniform1f(self.max_distance_loc, self.max_distance)

        # -------------------------
        # 取得點雲資料，支援 Live / PLY 切換
        # -------------------------
        glUniform1i(glGetUniformLocation(self.shader_program, "useUniformColor"), 0)
        if self.show_global:
        # ►► 只顯示 Global Map ◄◄
            with global_lock:
                pts_array = global_pts[:global_size].copy().astype(np.float32)
            total_pts = global_size
        elif self.use_live_data:
            # ► 即時模式：讀取環狀緩衝並套用最新 ArUco 變換
            with self.points_lock:
                pts_array = self.ring_buffer.get_recent_points(self.retention_seconds)
                total_pts = self.ring_buffer.size
            with global_transform_lock:
                Camera_T_Aruco = global_transform.copy()
            # LiDAR 到 ArUco 的複合變換
            self.Lidar_T_Aruco = np.linalg.inv(Cam_T_Lidar) @ Camera_T_Aruco
            pts_array, _ = self.get_latest_live_points()
            # 動態座標系 (ArUco → Camera & LiDAR)
            self.Word_Point      = np.eye(4, dtype=np.float32)
            self.Camera_Position = np.linalg.inv(Camera_T_Aruco)
            self.Lidar_Position  = self.Camera_Position @ Cam_T_Lidar
        else:
            # ► 檔案模式：顯示已載入的點雲與姿態，完全不動用 ArUco
            # 檔案模式：依滑桿過濾 timestamp
            mask   = np.logical_and(
                self.loaded_points[:,3] >= self.file_time_start,
                self.loaded_points[:,3] <= self.file_time_end
            )
            pts3d    = self.loaded_points[mask, :3].astype(np.float32)
            pts_array = pts3d
            total_pts = self.loaded_points.shape[0]
            # 直接用 PLY 裡的三組 4×4 矩陣
            self.Word_Point      = self.loaded_poses.get("world",  np.eye(4, dtype=np.float32))
            self.Camera_Position = self.loaded_poses.get("camera", np.eye(4, dtype=np.float32))
            self.Lidar_Position  = self.loaded_poses.get("lidar",  np.eye(4, dtype=np.float32))

            self.Lidar_T_Aruco   = None  # 不再需要 transform_point_cloud
            # #復原用
            # pts3d_lidar_base = transform_point_cloud(pts3d, np.linalg.inv(self.Lidar_Position))
            # pts_array = transform_point_cloud(pts3d_lidar_base, np.linalg.inv(self.Lidar_Position))
        # -------------------------
        # 繪製點雲
        # -------------------------
        num_points = pts_array.shape[0]
        if num_points > 0:
            pts_array = np.ascontiguousarray(pts_array, dtype=np.float32)
            glBindBuffer(GL_ARRAY_BUFFER, self.point_vbo)
            glBufferSubData(GL_ARRAY_BUFFER, 0, pts_array.nbytes, pts_array)
            glBindVertexArray(self.point_vao)
            glDrawArrays(GL_POINTS, 0, min(num_points, self.max_points))
            glBindVertexArray(0)

        # -------------------------
        # 繪製背景格線
        # -------------------------
        glUniform1i(glGetUniformLocation(self.shader_program, "useUniformColor"), 1)
        glBindVertexArray(self.grid_vao)
        glUniform4f(glGetUniformLocation(self.shader_program, "uColor"), 0.7, 0.7, 0.7, 1.0)
        glDrawArrays(GL_LINES, 0, self.grid_vertex_count)
        glBindVertexArray(0)

        # -------------------------
        # 繪製動態座標系 (World / Camera / LiDAR)
        # -------------------------
        glLineWidth(3.0)
        self.draw_axes_from_matrix(self.Word_Point,      scale=1,   colors=[(1,0,0),(0,1,0),(0,0,1)])
        self.draw_axes_from_matrix(self.Camera_Position, scale=1,   colors=[(1,0.5,0),(0.5,1,0),(1,1,0)])
        self.draw_axes_from_matrix(self.Lidar_Position,  scale=1.5, colors=[(0,1,1),(1,0,1),(0.5,0.5,1)])
        glLineWidth(1.0)

        # -------------------------
        # 繪製靜態原點軸 (Origin XYZ)
        # -------------------------
        glLineWidth(0.1)
        glBindVertexArray(self.axes_vao)
        # X 軸 (紅)
        glUniform4f(glGetUniformLocation(self.shader_program, "uColor"), 1.0, 0.0, 0.0, 1.0)
        glDrawArrays(GL_LINES, 0, 2)
        # Y 軸 (綠)
        glUniform4f(glGetUniformLocation(self.shader_program, "uColor"), 0.0, 1.0, 0.0, 1.0)
        glDrawArrays(GL_LINES, 2, 2)
        # Z 軸 (藍)
        glUniform4f(glGetUniformLocation(self.shader_program, "uColor"), 0.0, 0.0, 1.0, 1.0)
        glDrawArrays(GL_LINES, 4, 2)
        glBindVertexArray(0)
        glLineWidth(1.0)

        # -------------------------
        # 繪製 Global Map 中的 ArUco 標記座標系
        # -------------------------
        if self.show_global:
            glUniform1i(glGetUniformLocation(self.shader_program, "useUniformColor"), 1)
            glLineWidth(2.0)
            for name, mat in global_coords.items():
                # 只畫 marker_* 開頭的
                if name.startswith("marker_"):
                    # mat 已經是世界座標下的 marker 變換
                    self.draw_axes_from_matrix(mat, scale=0.5)
            glLineWidth(1.0)

        imgui.new_frame()
        imgui.begin("Control Panel", True)
        imgui.set_window_size(720, 0, condition=imgui.ONCE)

        # --------- 兩欄主面板 (不在欄內插入分隔線) ---------
        imgui.columns(2, "main_columns")

        # ================= 左欄 ==================
        imgui.begin_group()
        imgui.text("I/O & Network")
        imgui.push_item_width(125)
        _, self.tcp_host = imgui.input_text("TCP IP", self.tcp_host, 64)
        _, self.tcp_port = imgui.input_int("TCP Port", self.tcp_port)
        imgui.pop_item_width()

        imgui.spacing()
        imgui.text("Save Source:")
        if imgui.radio_button("Live", self.save_source == 0): self.save_source = 0
        imgui.same_line()
        if imgui.radio_button("Global", self.save_source == 1): self.save_source = 1

        imgui.spacing()
        imgui.text("Actions")
        imgui.push_item_width(170)
        if imgui.button("Open Saved PLY"):
            tk.Tk().withdraw()
            path = filedialog.askopenfilename(filetypes=[("PLY files", "*.ply")])
            if path:
                self.load_ply_with_pose(path)
                self.use_live_data = False
        if imgui.button("Save to .PLY"):
            if self.save_source == 0:
                self.save_live_to_ply()
            else:
                self.save_global_to_ply()
        if imgui.button("Clear Point Cloud"):
            with self.points_lock:
                self.ring_buffer.clear()
        if imgui.button("Restart TCP Conn"):
            self.restart_tcp_connection()
        if imgui.button("Update ArUco"):
            with latest_transform_lock, global_transform_lock:
                global_transform[:] = latest_transform_matrix.copy()
        if imgui.button("Exit App"):
            glfw.set_window_should_close(self.window, True)
        imgui.pop_item_width()

        imgui.spacing()
        imgui.text("Send Mode:")
        if imgui.radio_button("Live##send", self.send_mode == 0): self.send_mode = 0
        imgui.same_line()
        if imgui.radio_button("Global##send", self.send_mode == 1): self.send_mode = 1
        imgui.same_line()
        if imgui.radio_button("File##send", self.send_mode == 2): self.send_mode = 2

        imgui.end_group()

        imgui.next_column()

        # ================= 右欄 ==================
        imgui.begin_group()

        imgui.text("Settings")
        imgui.push_item_width(120)
        _, self.retention_seconds = imgui.slider_float("Storage", self.retention_seconds, 1.0, 30.0)
        _, self.max_distance = imgui.slider_float("Max dist.", self.max_distance, 1.0, 20.0)
        imgui.pop_item_width()
        imgui.spacing()

        imgui.text("Main Mode")
        changed, self.use_live_data = imgui.checkbox("Live Mode", self.use_live_data)
        _, self.use_inner_fence = imgui.checkbox("Enable Inner Fence", self.use_inner_fence)
        _, self.use_outer_fence = imgui.checkbox("Enable Outer Fence", self.use_outer_fence)
        imgui.spacing()

        imgui.text("Fence Setting")
        imgui.push_item_width(180)
        imgui.text("Outer Min:")
        _, self.fence_outer_min = imgui.input_float3("##outermin", *self.fence_outer_min)
        imgui.text("Outer Max:")
        _, self.fence_outer_max = imgui.input_float3("##outermax", *self.fence_outer_max)
        imgui.text("Min:")
        _, self.fence_min = imgui.input_float3("##min", *self.fence_min)
        imgui.text("Max:")
        _, self.fence_max = imgui.input_float3("##max", *self.fence_max)
        imgui.pop_item_width()

        imgui.end_group()

        # --------- 回到單欄，下方放 Global Map ---------
        imgui.columns(1)
        imgui.spacing()

        # ========== Global Map 可收合區塊 ==========
        if imgui.collapsing_header("Global Map", visible=True, flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
            imgui.spacing()
            changed, new_val = imgui.checkbox("Show Global##toggle", self.show_global)
            if changed:
                self.show_global = new_val
            if imgui.button("Open Saved PLY##global"):
                tk.Tk().withdraw()
                path = filedialog.askopenfilename(
                    filetypes=[("PLY files", "*.ply")],
                    title="選擇要加入全域地圖的 PLY 檔案"
                )
                if path:
                    self.load_ply_with_pose(path)
                    with global_lock:
                        global_size = 0
                    global_coords.clear()
                    add_to_global(self.loaded_points[:, :3].copy())
                    for n, m in self.loaded_poses.items():
                        add_global_coord(n, m)
            if imgui.button("Add to Global Map"):
                add_to_global(pts_array.copy())
                add_global_coord("world",  self.Word_Point)
                add_global_coord("camera", self.Camera_Position)
                add_global_coord("lidar",  self.Lidar_Position)
                for mid, w2m in GLOBAL_TO_MARKER.items():
                    if mid == 0:
                        continue
                    add_global_coord(f"marker_{mid}", w2m)
            if imgui.button("Clear Global Points"):
                with global_lock:
                    global_size = 0
                    global_pts[:] = 0
                    global_coords.clear()
            if imgui.button("Reset View"):
                self.rotation_x = self.rotation_y = 0.0
                self.pan_offset = np.array([0.0, 0.0, 0.0], np.float32)
                self.zoom = 20.0

            imgui.text(f"Accumulated points: {global_size:,}")

        imgui.separator()
        imgui.text(f"FPS: {imgui.get_io().framerate:.1f}  |  pts: {pts_array.shape[0]:,}  |  Total: {total_pts:,}")

        imgui.end()
        imgui.render()
        self.impl.render(imgui.get_draw_data())
        glfw.swap_buffers(self.window)


    def run(self):
        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            self.impl.process_inputs()
            self.update()
            self.render()
        self.cleanup()

    def cleanup(self):
        self.impl.shutdown()
        glfw.terminate()

    def save_points_to_ply_binary_task(self, points: np.ndarray, filename: str, header: str):
        try:
            with open(filename, 'wb') as f:
                f.write(header.encode('utf-8'))
                points.astype(np.float32).tofile(f)
            print(f"已儲存 {len(points)} 筆點雲（二進制格式），儲存至 {filename}")
        except Exception as e:
            print(f"儲存點雲資料時發生錯誤: {e}")

    def save_live_to_ply(self):
        with self.points_lock:
            pts = self.ring_buffer.get_recent_points(self.retention_seconds)  # Nx3
        if pts.shape[0] == 0:
            print("沒有可儲存的點雲資料")
            return

        # 套用當前 ArUco 轉換
        with global_transform_lock:
            Camera_T_Aruco = global_transform.copy()
        Lidar_T_Aruco = np.linalg.inv(Cam_T_Lidar) @ Camera_T_Aruco

        # 先轉 XYZ，再把 timestamp 併回
        pts_xyz = transform_point_cloud(pts, Lidar_T_Aruco)
        pts3, stamps = self.get_latest_live_points()
        data4 = np.hstack((pts3, stamps))

        # PLY header，新增 timestamp 屬性
        header_lines = [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {len(data4)}",
            "property float x",
            "property float y",
            "property float z",
            "property float timestamp",
        ]
        # 把相機/雷射/世界 pose 當 comment 寫入
        def matrix_to_comments(name, mat):
            flat = mat.flatten()
            return [f"comment {name}_{i} {v:.6f}" for i, v in enumerate(flat)]
        camera_pose = np.linalg.inv(Camera_T_Aruco)
        lidar_pose  = camera_pose @ Cam_T_Lidar
        world_pose  = np.eye(4, dtype=np.float32)
        header_lines += matrix_to_comments("camera", camera_pose)
        header_lines += matrix_to_comments("lidar",  lidar_pose)
        header_lines += matrix_to_comments("world",  world_pose)
        header_lines.append("end_header")
        header = "\n".join(header_lines) + "\n"

        # 寫檔執行緒
        threading.Thread(
            target=self.save_points_to_ply_binary_task,
            args=(data4, os.path.join("saved_ply",
                                    f"pointcloud_{time.strftime('%Y%m%d_%H%M%S')}_with_pose.ply"),
                header),
            daemon=True
        ).start()

    def save_global_to_ply(self):
        # 1) 取出 global_pts
        with global_lock:
            pts = global_pts[:global_size].copy()
        if pts.size == 0:
            print("Global Map 是空的，無點可存")
            return

        # 2) 統一 timestamp (這裡我用當下時間)
        stamps = np.full((pts.shape[0],1), time.time(), dtype=np.float32)
        data4 = np.hstack((pts, stamps))  # shape=(N,4)

        # 3) PLY header（加入所有坐標系 comment）
        header_lines = [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {len(data4)}",
            "property float x",
            "property float y",
            "property float z",
            "property float timestamp",
        ]

        # helper：把一個 4x4 矩陣展開成多行 comment
        def matrix_to_comments(name, mat):
            flat = mat.flatten()
            return [f"comment {name}_{i} {v:.6f}" for i, v in enumerate(flat)]

        # 把 global_coords 裡的每個坐標系都寫入
        for name, mat in global_coords.items():
            header_lines += matrix_to_comments(name, mat)

        header_lines.append("end_header")
        header = "\n".join(header_lines) + "\n"

        # 4) 檔名
        filename = os.path.join(
            "saved_ply",
            f"global_{time.strftime('%Y%m%d_%H%M%S')}.ply"
        )

        # 5) 背景執行緒寫入
        threading.Thread(
            target=self.save_points_to_ply_binary_task,
            args=(data4, filename, header),
            daemon=True
        ).start()
        print(f"開始將 Global Map (含所有坐標系) 寫入 {filename}…")


 # -------------------------
# 額外：繪製 ArUco、Camera、LiDAR 的動態坐標系（XYZ）
# -------------------------
    def draw_axes_from_matrix(self, matrix, scale=0.2, colors=None):
        if colors is None:
            colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # 預設 RGB

        key = tuple(matrix.flatten()) + (scale,)
        vao, vertex_count = self._axis_vao_cache.get(key, (None, 0))

        if vao is None:
            origin = matrix[:3, 3]
            x_axis = origin + matrix[:3, 0] * scale
            y_axis = origin + matrix[:3, 1] * scale
            z_axis = origin + matrix[:3, 2] * scale

            vertices = np.array([
                *origin, *x_axis,
                *origin, *y_axis,
                *origin, *z_axis
            ], dtype=np.float32)

            vao = glGenVertexArrays(1)
            vbo = glGenBuffers(1)
            glBindVertexArray(vao)
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
            glBindVertexArray(0)
            vertex_count = 6
            self._axis_vao_cache[key] = (vao, vertex_count)

        glUseProgram(self.shader_program)
        glBindVertexArray(vao)
        glLineWidth(3.0)
        glUniform1i(glGetUniformLocation(self.shader_program, "useUniformColor"), 1)
        glUniform4f(glGetUniformLocation(self.shader_program, "uColor"), *colors[0], 1)
        glDrawArrays(GL_LINES, 0, 2)
        glUniform4f(glGetUniformLocation(self.shader_program, "uColor"), *colors[1], 1)
        glDrawArrays(GL_LINES, 2, 2)
        glUniform4f(glGetUniformLocation(self.shader_program, "uColor"), *colors[2], 1)
        glDrawArrays(GL_LINES, 4, 2)
        glBindVertexArray(0)
        glLineWidth(1.0)


    def load_ply_with_pose(self, filepath: str):
        if not os.path.exists(filepath):
            messagebox.showerror("錯誤", f"找不到檔案：{filepath}")
            return

        with open(filepath, 'rb') as f:
            header = []
            properties = []
            while True:
                line = f.readline().decode('utf-8').strip()
                header.append(line)
                # 收集所有 property float <name>
                if line.startswith("property"):
                    parts = line.split()
                    if len(parts) == 3 and parts[1] == "float":
                        properties.append(parts[2])
                if line == "end_header":
                    break

            # 讀頂點數
            vertex_cnt = next(int(l.split()[-1])
                              for l in header if l.startswith("element vertex"))

            # 動態解析所有 comment，包含各種 <name>_<idx>
            poses_dict = {}
            for l in header:
                if not l.startswith("comment"):
                    continue
                _, tag, val = l.split()
                # tag 格式：<name>_<idx>
                name, idx_str = tag.rsplit("_", 1)
                idx = int(idx_str)
                poses_dict.setdefault(name, [0.0] * 16)
                poses_dict[name][idx] = float(val)

            # 建立 loaded_poses：name → 4×4 矩陣
            self.loaded_poses = {
                name: np.array(vals, dtype=np.float32).reshape(4, 4)
                for name, vals in poses_dict.items()
            }

            # 根據 property 數量動態讀取 data
            prop_cnt = len(properties)  # 3 或 4
            raw = np.fromfile(f, dtype=np.float32, count=vertex_cnt * prop_cnt)
            pts = raw.reshape(-1, prop_cnt)

            # 如果只有 x,y,z，就補一欄 timestamp=0
            if prop_cnt == 3 or "timestamp" not in properties:
                xyz = pts[:, :3]
                t   = np.zeros((vertex_cnt,1), dtype=np.float32)
                all_pts = np.hstack((xyz, t))
                has_time = False
            else:
                idx_t = properties.index("timestamp")
                xyz = pts[:, :3]
                t   = pts[:, idx_t].reshape(-1,1)
                all_pts = np.hstack((xyz, t))
                has_time = True

            self.loaded_points = all_pts.astype(np.float32)

            # 初始化時間範圍
            if has_time:
                self.file_time_min   = float(all_pts[:,3].min())
                self.file_time_max   = float(all_pts[:,3].max())
            else:
                self.file_time_min   = 0.0
                self.file_time_max   = 0.0

            self.file_time_start = self.file_time_min
            self.file_time_end   = self.file_time_max

        print(f"✅ 載入 {vertex_cnt} 點，"
              f"{'含' if has_time else '不含'} timestamp，"
              f"時間範圍：{self.file_time_min:.3f} – {self.file_time_max:.3f}，"
              f"解析到坐標系：{list(self.loaded_poses.keys())}")

class ArUcoTransformReceiver(threading.Thread):
    def __init__(self, udp_port=9002):
        super().__init__(daemon=True)
        self.port = udp_port

    def run(self):
        global global_transform
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", self.port))
        print(f"ArUco 接收埠開啟：UDP {self.port}")
        while True:
            try:
                data, _ = sock.recvfrom(4096)
                obj = json.loads(data.decode("utf-8"))
                matrix = np.array(obj["T"], dtype=np.float32).reshape((4, 4))
                with global_transform_lock:
                    latest_transform_matrix[:] = matrix
                print(f"✅ 接收到 ArUco: marker {obj['id']}")
            except Exception as e:
                print(f"❌ 解析 ArUco UDP 失敗: {e}")

import cv2
import threading

def gstreamer_receiver_thread(stop_event=None):
    gst_str = (
        'udpsrc port=5000 caps="application/x-rtp, media=video, '
        'clock-rate=90000, encoding-name=H264" ! '
        'rtph264depay ! avdec_h264 ! videoconvert ! appsink drop=1 sync=false'
    )

    cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("❌ 無法開啟 GStreamer 視訊串流")
        return

    print("✅ 開始接收 GStreamer 串流（按下 Q 或 ESC 離開）")
    
    try:
        while stop_event is None or not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("⚠️ 無法讀取串流影像")
                continue

            # 可根據需要縮放顯示影像
            # frame = cv2.resize(frame, (640, 480))
            cv2.imshow("Camera Stream", frame)

            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):  # ESC 或 q 鍵
                print("🛑 使用者中止串流")
                break
    except Exception as e:
        print(f"❌ 發生例外錯誤：{e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()



# =============================================================================
# 程式進入點：
# 先啟動 ArUco 偵測線程，再建立並運行點雲視覺化應用
# =============================================================================
if __name__ == '__main__':
    # aruco_thread = threading.Thread(target=detect_aruco_thread, args=(0,), daemon=True)
    # aruco_thread.start()
    # ✅ 啟動遠端 UDP ArUco 接收線程))
    aruco_recv_thread = ArUcoTransformReceiver(udp_port=9002)
    aruco_recv_thread.start()
    threading.Thread(target=gstreamer_receiver_thread, daemon=True).start()
    live_view   = PointCloudViewer()

    while not glfw.window_should_close(live_view.window):
        glfw.poll_events()
        glfw.make_context_current(live_view.window)
        live_view.impl.process_inputs()
        live_view.update()
        live_view.render()

    glfw.terminate()


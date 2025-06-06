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

        if ids is not None:
            frame = aruco.drawDetectedMarkers(frame, corners, ids)
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.19, camera_matrix, dist_coeffs)
            # 取第一個標記作為示例
            rvec = rvecs[0][0]
            tvec = tvecs[0][0]
            # 旋轉向量轉換為旋轉矩陣
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            # 組合成 4x4 齊次變換矩陣
            transform_matrix = np.eye(4, dtype=np.float32)
            transform_matrix[:3, :3] = rotation_matrix.astype(np.float32)
            transform_matrix[:3, 3] = tvec.astype(np.float32)
            # 更新全域變數
            with latest_transform_lock:
                 latest_transform_matrix[:] = transform_matrix.copy()

            # 顯示 3D 坐標軸於影像上（多個標記時依序顯示）
            for rv, tv in zip(rvecs, tvecs):
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rv, tv, 0.1)

        cv2.imshow('ArUco Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# =============================================================================
# 點雲轉換函數：利用 4x4 齊次變換矩陣轉換 (N, 3) 點雲資料
# =============================================================================
def transform_point_cloud(points, transform_matrix):
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
    transformed = (transform_matrix @ points_hom.T).T
    return transformed[:, :3]

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

# 平移矩陣
T = np.eye(4, dtype=np.float32)
T[:3, 3] = [dx, dy, dz]

# Camera to LiDAR 的外參（先旋轉再平移）
Cam_T_Lidar = T @ Rz

# =============================================================================
# PointRingBuffer 類別：環狀點雲資料緩衝區
# =============================================================================
class PointRingBuffer:
    def __init__(self, max_points):
        self.max_points = max_points
        self.buffer = np.zeros((max_points, 4), dtype=np.float64)  # 存 (x,y,z,timestamp)
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

    def get_recent_points(self, retention_time):
        if self.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        now = time.time()
        cutoff = now - retention_time
        if self.size < self.max_points:
            data = self.buffer[:self.size]
        else:
            data = np.roll(self.buffer, -self.index, axis=0)
        idx = np.searchsorted(data[:, 3], cutoff, side='left')
        valid = data[idx:, :3]
        return valid.astype(np.float32)

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
            pts = np.array(floats, dtype=np.float64).reshape(-1, 3)
            t_now = time.time()
            timestamps = np.full((pts.shape[0], 1), t_now, dtype=np.float64)
            new_points = np.hstack((pts, timestamps))
            with self.lock:
                self.ring_buffer.add_points(new_points)


# 🔧 自訂 JSON 編碼器：讓 numpy 的矩陣與數值可以轉成 JSON
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # 把 numpy 矩陣轉成 list
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
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
    def __init__(self, width=1600, height=800):
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
        
        # 啟動 UDP 接收線程
        self.udp_receiver = UDPReceiver('0.0.0.0', 8080, self.ring_buffer, self.points_lock)
        self.udp_receiver.start()

        # 初始化 GLFW、ImGui、Shader 與 OpenGL 緩衝區
        self.init_glfw()
        self.init_imgui()
        self.init_shaders()
        self.init_buffers()

        # 啟動 TCP 傳送（此處示例連線至指定的 IP 與埠）
        self.tcp_sender_thread = threading.Thread(target=self.tcp_sender, args=("192.168.137.1", 9000), daemon=True)
        #self.tcp_sender_thread = threading.Thread(target=self.tcp_sender, args=("127.0.0.1", 9000), daemon=True)
        self.tcp_sender_thread.start()
        # 定義預設的坐標系資訊，以 identity matrix 作為範例
        self.coordinate_system = {
            "matrix": list(pyrr.matrix44.create_identity(dtype=np.float32))
        }

        self.Word_Point = None
        self.Camera_Position = None 
        self.Lidar_Position =  None
        self.Lidar_T_Aruco = None
    

    def tcp_sender(self, host, port):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((host, port))
            print(f"TCP Sender: 連線到 {host}:{port}")
        except Exception as e:
            print(f"TCP Sender 無法連線: {e}")
            return

        while True:
            with self.points_lock:
                raw_points = self.ring_buffer.get_recent_points(self.retention_seconds)
                transformed = transform_point_cloud(raw_points, self.Lidar_T_Aruco)  # or any matrix you want
                points_bin = transformed.astype(np.float32).tobytes()

            # ✅ 使用自訂編碼器處理 numpy 物件
            coord_system = {
                "lidar":  self.Lidar_Position,
                "camera": self.Camera_Position,
                "world":  self.Word_Point
            }
            coord_json = json.dumps(coord_system, cls=NumpyEncoder)
            coord_bin = coord_json.encode('utf-8')

            header = struct.pack('<II', len(coord_bin), len(points_bin))
            message = header + coord_bin + points_bin

            try:
                sock.sendall(message)
            except Exception as e:
                print(f"TCP Sender 傳送錯誤: {e}")
                break

            time.sleep(self.retention_seconds)





    def restart_tcp_connection(self, host, port):
        if not hasattr(self, 'tcp_sender_thread') or not self.tcp_sender_thread.is_alive():
            print("TCP 傳送線程已停止，正在重啟...")
            self.tcp_sender_thread = threading.Thread(target=self.tcp_sender, args=(host, port), daemon=True)
            self.tcp_sender_thread.start()
        else:
            print("TCP 傳送線程仍在運行。")

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
        imgui.create_context()
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
        self.projection = pyrr.matrix44.create_perspective_projection_matrix(
            45, width / height, 0.1, 100.0
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
        glClearColor(0.2, 0.2, 0.2, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # 建立 view 矩陣
        base_eye = np.array([0.0, -self.zoom, 5.0], dtype=np.float32)
        base_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        eye = base_eye + self.pan_offset
        target = base_target + self.pan_offset
        view = pyrr.matrix44.create_look_at(eye, target, np.array([0.0, 1.0, 0.0], dtype=np.float32))

        # 建立模型矩陣（旋轉）
        pitch = pyrr.matrix44.create_from_x_rotation(self.rotation_x)
        yaw = pyrr.matrix44.create_from_z_rotation(self.rotation_y)
        model = pyrr.matrix44.multiply(yaw, pitch)

        MVP = pyrr.matrix44.multiply(model, view)
        MVP = pyrr.matrix44.multiply(MVP, self.projection)
        self.last_model, self.last_view, self.last_projection = model, view, self.projection

        glUseProgram(self.shader_program)
        glUniformMatrix4fv(self.mvp_loc, 1, GL_FALSE, MVP)
        glUniform1f(self.max_distance_loc, self.max_distance)

        # -------------------------
        # 取得點雲資料，並利用全域 ArUco 變換與相機到光達外參作複合變換
        # -------------------------
        glUniform1i(glGetUniformLocation(self.shader_program, "useUniformColor"), 0)
        with self.points_lock:
            pts_array = self.ring_buffer.get_recent_points(self.retention_seconds)
            total_pts = self.ring_buffer.size

        # 取得 ArUco 變換矩陣（線程安全）
        with global_transform_lock:
            Camera_T_Aruco = global_transform.copy()
        # 複合變換：Lidar_T_Aruco = T_aruco @ T_cam2lidar
        self.Lidar_T_Aruco = np.linalg.inv(Cam_T_Lidar) @ Camera_T_Aruco 
        pts_array = transform_point_cloud(pts_array, self.Lidar_T_Aruco)

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
        # 繪製 XYZ 軸線
        # -------------------------
        glLineWidth(0.1)  
        glBindVertexArray(self.axes_vao)
        glUniform4f(glGetUniformLocation(self.shader_program, "uColor"), 1.0, 0.0, 0.0, 1.0)
        glDrawArrays(GL_LINES, 0, 2)
        glUniform4f(glGetUniformLocation(self.shader_program, "uColor"), 0.0, 1.0, 0.0, 1.0)
        glDrawArrays(GL_LINES, 2, 2)
        glUniform4f(glGetUniformLocation(self.shader_program, "uColor"), 0.0, 0.0, 1.0, 1.0)
        glDrawArrays(GL_LINES, 4, 2)
        glBindVertexArray(0)

                # 取得轉換矩陣
        with global_transform_lock:
            Camera_T_Aruco = global_transform.copy()

       
        self.Word_Point = np.eye(4)
        self.Camera_Position = self.Word_Point @ np.linalg.inv(Camera_T_Aruco)  # ArUco 轉換矩陣的逆矩陣
        self.Lidar_Position = self.Camera_Position @ Cam_T_Lidar  # LiDAR 轉換矩陣的逆矩陣
        # 使用與固定 XYZ 相同顏色繪製三組動態座標軸

        self.draw_axes_from_matrix(self.Word_Point, scale=1, colors=[(1, 0, 0), (0, 1, 0), (0, 0, 1)])       # ArUco 用 R,G,B
        self.draw_axes_from_matrix(self.Camera_Position, scale=1, colors=[(1, 0.5, 0), (0.5, 1, 0), (1, 1, 0)])  # Camera 用 橙, 淺綠, 黃
        self.draw_axes_from_matrix(self.Lidar_Position, scale=1.5, colors=[(0, 1, 1), (1, 0, 1), (0.5, 0.5, 1)])   # LiDAR 用 青, 紫, 淡藍


        glLineWidth(1.0)  # <=== 繪製完要設回來
        # -------------------------
        # ImGui 控制面板
        # -------------------------
        imgui.new_frame()
        imgui.begin("Control Panel")
        _, self.retention_seconds = imgui.slider_float("Storage seconds", self.retention_seconds, 1.0, 30.0)
        _, self.max_distance = imgui.slider_float("Max distance", self.max_distance, 1.0, 20.0)
        if imgui.button("Clear Point Cloud"):
            with self.points_lock:
                self.ring_buffer.clear()
        if imgui.button("Save to .PLY"):
            self.save_points_to_ply()
        if imgui.button("Restart TCP Connection"):
            self.restart_tcp_connection("192.168.137.1", 9000)
            #self.restart_tcp_connection("127.0.0.1", 9000)
        if imgui.button("Reset View"):
            self.rotation_x = 0.0
            self.rotation_y = 0.0
            self.pan_offset = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self.zoom = 20.0
        if imgui.button("Exit Application"):
            glfw.set_window_should_close(self.window, True)
        if imgui.button("Update ArUco Transform"):
            with latest_transform_lock:
                with global_transform_lock:
                    global_transform[:] = latest_transform_matrix.copy()
            print("✅ 已套用最新 ArUco 位置至 global_transform")
        imgui.text(f"Current points: {pts_array.shape[0]}")
        imgui.text(f"Total stored: {total_pts}")
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

    def save_points_to_ply(self):
        with self.points_lock:
            raw_points = self.ring_buffer.get_recent_points(self.retention_seconds)

        if raw_points.size == 0:
            print("沒有可儲存的點雲資料")
            return

        # 套用當前姿態轉換
        with global_transform_lock:
            Camera_T_Aruco = global_transform.copy()
        Lidar_T_Aruco = np.linalg.inv(Cam_T_Lidar) @ Camera_T_Aruco
        transformed = transform_point_cloud(raw_points, Lidar_T_Aruco)

        # 儲存的姿態矩陣
        camera_pose = np.linalg.inv(Camera_T_Aruco)
        lidar_pose = camera_pose @ Cam_T_Lidar
        world_pose = np.eye(4, dtype=np.float32)

        # ✅ 建立資料夾
        save_dir = "saved_ply"
        os.makedirs(save_dir, exist_ok=True)

        filename = os.path.join(save_dir, f"pointcloud_{time.strftime('%Y%m%d_%H%M%S')}_with_pose.ply")

        # 建立 PLY header + comment 標記姿態資訊
        header_lines = [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {len(transformed)}",
            "property float x",
            "property float y",
            "property float z",
        ]

        def matrix_to_comments(name, mat):
            flat = mat.flatten()
            return [f"comment {name}_{i} {v:.6f}" for i, v in enumerate(flat)]

        header_lines += matrix_to_comments("camera", camera_pose)
        header_lines += matrix_to_comments("lidar", lidar_pose)
        header_lines += matrix_to_comments("world", world_pose)
        header_lines.append("end_header")
        header = "\n".join(header_lines) + "\n"

        # 啟動儲存執行緒
        threading.Thread(target=self.save_points_to_ply_binary_task,
                        args=(transformed, filename, header),
                        daemon=True).start()

 # -------------------------
# 額外：繪製 ArUco、Camera、LiDAR 的動態坐標系（XYZ）
# -------------------------
    def draw_axes_from_matrix(self, matrix, scale=0.2, colors=None):
        if colors is None:
            colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # 預設 RGB

        glUseProgram(self.shader_program)
        origin = matrix[:3, 3]
        x_axis = origin + matrix[:3, 0] * scale
        y_axis = origin + matrix[:3, 1] * scale
        z_axis = origin + matrix[:3, 2] * scale

        vertices = np.array([
            *origin, *x_axis,  # X 軸
            *origin, *y_axis,  # Y 軸
            *origin, *z_axis   # Z 軸
        ], dtype=np.float32)

        vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

        glLineWidth(3.0)
        glUniform1i(glGetUniformLocation(self.shader_program, "useUniformColor"), 1)

        # 為每條軸設置不同顏色
        glUniform4f(glGetUniformLocation(self.shader_program, "uColor"), *colors[0], 1)
        glDrawArrays(GL_LINES, 0, 2)
        glUniform4f(glGetUniformLocation(self.shader_program, "uColor"), *colors[1], 1)
        glDrawArrays(GL_LINES, 2, 2)
        glUniform4f(glGetUniformLocation(self.shader_program, "uColor"), *colors[2], 1)
        glDrawArrays(GL_LINES, 4, 2)

        glBindVertexArray(0)
        glDeleteVertexArrays(1, [vao])
        glDeleteBuffers(1, [vbo])
        glLineWidth(1.0)



# =============================================================================
# 程式進入點：
# 先啟動 ArUco 偵測線程，再建立並運行點雲視覺化應用
# =============================================================================
if __name__ == '__main__':
    # 啟動 ArUco 偵測
    aruco_thread = threading.Thread(target=detect_aruco_thread, args=(0,), daemon=True)
    aruco_thread.start()

    # 啟動點雲視覺化
    viewer = PointCloudViewer()
    viewer.run()

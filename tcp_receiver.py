import glfw                       # 建立與管理視窗及處理輸入
import threading
from OpenGL.GL import *           # OpenGL API，用於實際渲染圖形
import numpy as np                # 高效數值計算與陣列操作
import pyrr                       # 提供矩陣、向量及旋轉計算
import socket                     # 用於網路通訊
import struct                     # 用於封包中數據的編解碼
import time                       # 計時與取得時間戳
import imgui                      # 即時 GUI 套件
import json
from imgui.integrations.glfw import GlfwRenderer

# =============================================================================
# PointRingBuffer 類別：環狀緩衝區
# -----------------------------------------------------------------------------
# 用於固定大小內儲存點雲資料與時間戳，當空間滿時自動覆蓋最舊的資料
# =============================================================================
class PointRingBuffer:
    def __init__(self, max_points):
        self.max_points = max_points
        # 每筆資料包含 (x, y, z, timestamp)，預設使用 float64
        self.buffer = np.zeros((max_points, 4), dtype=np.float64)
        self.index = 0
        self.size = 0

    def add_points(self, new_points: np.ndarray):
        if new_points is None or new_points.size == 0:
            return
        if new_points.shape[1] != 4:
            raise ValueError("新加入的點必須為 Nx4 陣列（x, y, z, timestamp）")
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
            return np.empty((0, 3), dtype=np.float32)
        now = time.monotonic()
        cutoff = now - retention_time

        if self.size < self.max_points:
            data = self.buffer[:self.size]
        else:
            # 利用 np.roll 將環狀緩衝區卷成連續的陣列
            data = np.roll(self.buffer, -self.index, axis=0)

        # 使用二分搜尋取得符合保留時間條件的點（只取 x,y,z）
        idx = np.searchsorted(data[:, 3], cutoff, side='left')
        valid = data[idx:, :3]
        # print(">>> [DEBUG] 現在時間:", time.time())
        # print(">>> [DEBUG] 資料 timestamp 前幾筆:", data[:5, 3])
        # print(">>> [DEBUG] retention_time 設定為:", retention_time)
        # print(">>> [DEBUG] cutoff 時間 = ", cutoff)
        # print(">>> [DEBUG] 通過條件的點數 = ", valid.shape[0])
        return valid.astype(np.float32)
    


    def clear(self):
        self.buffer[:] = 0
        self.index = 0
        self.size = 0

# =============================================================================
# Shader 工具函式
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

# -----------------------------------------------------------------------------
# Vertex Shader：負責點雲頂點的轉換與計算到原點的距離
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Fragment Shader：根據距離資訊使用 Jet colormap 進行著色
# -----------------------------------------------------------------------------
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
# 輔助函式：確保 TCP socket 讀取指定數量位元組
# =============================================================================
def recvall(sock, n):
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

# =============================================================================
# TCPReceiver 類別：透過 TCP 接收點雲資料並存入環狀緩衝區
# =============================================================================
class TCPReceiver(threading.Thread):
    def __init__(self, host, port, ring_buffer, lock, viewer=None):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.ring_buffer = ring_buffer
        self.lock = lock
        self.viewer = viewer
        self._stop_event = threading.Event()
        self.server_socket = None

    def run(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        print(f"TCP Receiver 監聽 {self.host}:{self.port}")

        while not self._stop_event.is_set():
            try:
                self.server_socket.settimeout(1.0)
                conn, addr = self.server_socket.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            print(f"TCP Receiver 已連線：{addr}")
            try:
                while not self._stop_event.is_set():
                    header = self._recvall(conn, 8)
                    if header is None:
                        break
                    coord_len, pts_len = struct.unpack('<II', header)
                    coord_bytes = self._recvall(conn, coord_len)
                    if coord_bytes is None:
                        break
                    try:
                        coord_sys = json.loads(coord_bytes.decode('utf-8'))
                    except Exception as e:
                        print(f"坐標系統資訊解碼錯誤: {e}")
                        continue

                    pts_bytes = self._recvall(conn, pts_len)
                    if pts_bytes is None:
                        break
                    pts4 = np.frombuffer(pts_bytes, dtype=np.float32).reshape(-1, 4).astype(np.float64)
                    with self.lock:
                        self.ring_buffer.add_points(pts4)

                    if self.viewer:
                        if self.viewer.remote_coordinate_system is None:
                            self.viewer.remote_coordinate_system = {}
                        with self.viewer.coord_lock:  # 👈 加這行
                            for k, v in coord_sys.items():
                                arr = np.array(v, dtype=np.float32)
                                if arr.shape == (4, 4):
                                    self.viewer.remote_coordinate_system[k] = arr

                    glfw.post_empty_event()
            except Exception as e:
                print(f"TCP 接收錯誤：{e}")
            finally:
                conn.close()
                print("TCP Receiver: client 已斷線，等待下次連線…")

    def stop(self):
        self._stop_event.set()
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception:
                pass

    @staticmethod
    def _recvall(sock, n):
        data = b''
        while len(data) < n:
            try:
                packet = sock.recv(n - len(data))
            except socket.error:
                return None
            if not packet:
                return None
            data += packet
        return data


# =============================================================================
# PointCloudViewer 類別：OpenGL 繪製與使用者互動
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

        # 啟動 TCPReceiver，監聽 0.0.0.0:9000 接收點雲資料
        self.remote_coordinate_system = None
        self.tcp_receiver = TCPReceiver('0.0.0.0', 9000, self.ring_buffer, self.points_lock)
        self.tcp_receiver.viewer = self
        self.tcp_receiver.start()
        self.coord_lock = threading.Lock()

        self.init_glfw()
        self.init_imgui()
        self.init_shaders()
        self.init_buffers()
        self.remote_coordinate_system = None  # 儲存接收到的 matrix (4x4)
        self.tcp_receiver.viewer = self


    def init_glfw(self):
        if not glfw.init():
            raise Exception("GLFW 初始化失敗")
        self.window = glfw.create_window(self.width, self.height, "TCP Point Cloud Viewer", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("無法建立視窗")
        glfw.make_context_current(self.window)
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_cursor_pos_callback(self.window, self.cursor_pos_callback)
        glEnable(GL_DEPTH_TEST)

    def init_imgui(self):
        imgui.create_context()
        self.impl = GlfwRenderer(self.window)

    def init_shaders(self):
        self.shader_program = create_shader_program(vertex_shader_source, fragment_shader_source)
        self.mvp_loc = glGetUniformLocation(self.shader_program, "MVP")
        self.max_distance_loc = glGetUniformLocation(self.shader_program, "maxDistance")
        self.max_distance = 5.0

    def init_buffers(self):
        self.max_points = 19200000  # 渲染時最大點數
        self.point_vao = glGenVertexArrays(1)
        self.point_vbo = glGenBuffers(1)
        glBindVertexArray(self.point_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.point_vbo)
        glBufferData(GL_ARRAY_BUFFER, self.max_points * 3 * 4, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glBindVertexArray(0)

        # 建立格線
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
            0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 1
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
            self.rotation_y += (xpos - last_x) * sensitivity
            self.rotation_x += (ypos - last_y) * sensitivity
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
        if glfw.get_key(self.window, glfw.KEY_A) == glfw.PRESS:
            self.pan_offset[0] -= pan_step
        if glfw.get_key(self.window, glfw.KEY_D) == glfw.PRESS:
            self.pan_offset[0] += pan_step
        if glfw.get_key(self.window, glfw.KEY_W) == glfw.PRESS:
            self.pan_offset[1] += pan_step
        if glfw.get_key(self.window, glfw.KEY_S) == glfw.PRESS:
            self.pan_offset[1] -= pan_step
        if glfw.get_key(self.window, glfw.KEY_Q) == glfw.PRESS:
            self.pan_offset[2] -= pan_step
        if glfw.get_key(self.window, glfw.KEY_E) == glfw.PRESS:
            self.pan_offset[2] += pan_step
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

    def draw_axes_from_matrix(self, matrix, scale=1.0, colors=None):
        if colors is None:
            colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # RGB

        glUseProgram(self.shader_program)
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

        glLineWidth(3.0)
        glUniform1i(glGetUniformLocation(self.shader_program, "useUniformColor"), 1)
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
    
    def restart_tcp_receiver(self, host="0.0.0.0", port=9000):
        if hasattr(self, 'tcp_receiver') and self.tcp_receiver.is_alive():
            print("正在停止舊的 TCP Receiver...")
            self.tcp_receiver.stop()
            self.tcp_receiver.join()
            print("舊的 TCP Receiver 已停止")

        print("重新啟動 TCP Receiver 中...")
        self.tcp_receiver = TCPReceiver(host, port, self.ring_buffer, self.points_lock, viewer=self)
        self.tcp_receiver.start()
        print("新的 TCP Receiver 已啟動")




    def render(self):
        glClearColor(0.2, 0.2, 0.2, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        base_eye = np.array([0.0, -self.zoom, 5.0], dtype=np.float32)
        base_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        eye = base_eye + self.pan_offset
        target = base_target + self.pan_offset
        view = pyrr.matrix44.create_look_at(eye, target, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        pitch = pyrr.matrix44.create_from_x_rotation(self.rotation_x)
        yaw = pyrr.matrix44.create_from_z_rotation(self.rotation_y)
        model = pyrr.matrix44.multiply(yaw, pitch)
        # flip = pyrr.matrix44.create_from_x_rotation(np.pi)
        # model = pyrr.matrix44.multiply(model, flip)
        MVP = pyrr.matrix44.multiply(model, view)
        MVP = pyrr.matrix44.multiply(MVP, self.projection)
        self.last_model, self.last_view, self.last_projection = model, view, self.projection

        glUseProgram(self.shader_program)
        glUniformMatrix4fv(self.mvp_loc, 1, GL_FALSE, MVP)
        glUniform1f(self.max_distance_loc, self.max_distance)

        # 繪製點雲
        glUniform1i(glGetUniformLocation(self.shader_program, "useUniformColor"), 0)
        with self.points_lock:
            pts_array = self.ring_buffer.get_recent_points(self.retention_seconds)
            total_pts = self.ring_buffer.size
        num_points = pts_array.shape[0]
        if num_points > 0:
            pts_array = np.ascontiguousarray(pts_array, dtype=np.float32)
            glBindBuffer(GL_ARRAY_BUFFER, self.point_vbo)
            glBufferSubData(GL_ARRAY_BUFFER, 0, pts_array.nbytes, pts_array)
            glBindVertexArray(self.point_vao)
            glDrawArrays(GL_POINTS, 0, min(num_points, self.max_points))
            glBindVertexArray(0)

        if self.remote_coordinate_system:
            with self.coord_lock:
                for name, matrix in self.remote_coordinate_system.items():
                    if name == "lidar":
                        self.draw_axes_from_matrix(matrix, scale=1.5, colors=[(0, 1, 1), (1, 0, 1), (0.5, 0.5, 1)])  # 青紫藍
                    elif name == "camera":
                        self.draw_axes_from_matrix(matrix, scale=1.0, colors=[(1, 0.5, 0), (0.5, 1, 0), (1, 1, 0)])  # 橘綠黃
                    elif name == "world":
                        self.draw_axes_from_matrix(matrix, scale=1.0, colors=[(1, 0, 0), (0, 1, 0), (0, 0, 1)])  # 紅綠藍
                    elif name.startswith("marker_"):
                        self.draw_axes_from_matrix(matrix, scale=0.6, colors=[(0.6, 0.6, 0.6), (0.3, 0.3, 0.3), (0.2, 0.2, 0.2)])  # 淺灰系



        # 繪製格線
        glUniform1i(glGetUniformLocation(self.shader_program, "useUniformColor"), 1)
        glBindVertexArray(self.grid_vao)
        glUniform4f(glGetUniformLocation(self.shader_program, "uColor"), 0.7, 0.7, 0.7, 1.0)
        glDrawArrays(GL_LINES, 0, self.grid_vertex_count)
        glBindVertexArray(0)

        # 繪製 XYZ 軸線
        glBindVertexArray(self.axes_vao)
        glUniform4f(glGetUniformLocation(self.shader_program, "uColor"), 1.0, 0.0, 0.0, 1.0)
        glDrawArrays(GL_LINES, 0, 2)
        glUniform4f(glGetUniformLocation(self.shader_program, "uColor"), 0.0, 1.0, 0.0, 1.0)
        glDrawArrays(GL_LINES, 2, 2)
        glUniform4f(glGetUniformLocation(self.shader_program, "uColor"), 0.0, 0.0, 1.0, 1.0)
        glDrawArrays(GL_LINES, 4, 2)
        glBindVertexArray(0)

        # 繪製 GUI 控制面板（可用來調整保留時間、最大距離等參數）
        imgui.new_frame()
        imgui.begin("Control Panel")
        _, self.retention_seconds = imgui.slider_float("Storage seconds", self.retention_seconds, 1.0, 30.0)
        _, self.max_distance = imgui.slider_float("Max distance", self.max_distance, 1.0, 20.0)
        if imgui.button("Clear Point Cloud"):
            with self.points_lock:
                self.ring_buffer.clear()
        if imgui.button("Reset View"):
            self.rotation_x = 0.0
            self.rotation_y = 0.0
            self.pan_offset = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self.zoom = 20.0
        if imgui.button("Restart TCP Receiver"):
            self.restart_tcp_receiver()
        if imgui.button("Exit Application"):
            glfw.set_window_should_close(self.window, True)
            
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

if __name__ == '__main__':
    viewer = PointCloudViewer()
    viewer.run()

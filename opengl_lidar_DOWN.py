import glfw
from OpenGL.GL import *
import numpy as np
import pyrr
import socket
import struct
import threading
import time
import imgui
from imgui.integrations.glfw import GlfwRenderer
from scipy.spatial import cKDTree  # 放到最上面 import 區
# -------------------------------
# Ring Buffer 類別：高效儲存點雲資料與時間戳
# -------------------------------
class PointRingBuffer:
    def __init__(self, max_points):
        self.max_points = max_points
        self.buffer = np.zeros((max_points, 4), dtype=np.float64)  # x, y, z, timestamp
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
# Shader 工具函式與原始碼
# -------------------------------
def compile_shader(shader_type, source):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
        error = glGetShaderInfoLog(shader)
        raise RuntimeError("Shader compile error: " + error.decode())
    return shader

def create_shader_program(vertex_src, fragment_src):
    vs = compile_shader(GL_VERTEX_SHADER, vertex_src)
    fs = compile_shader(GL_FRAGMENT_SHADER, fragment_src)
    program = glCreateProgram()
    glAttachShader(program, vs)
    glAttachShader(program, fs)
    glLinkProgram(program)
    if glGetProgramiv(program, GL_LINK_STATUS) != GL_TRUE:
        error = glGetProgramInfoLog(program)
        raise RuntimeError("Shader link error: " + error.decode())
    glDeleteShader(vs)
    glDeleteShader(fs)
    return program

vertex_shader_source = """
#version 330 core
layout(location=0) in vec3 aPos;

uniform mat4 MVP;
out float vDistance;

void main(){
    vDistance = length(aPos);  // 計算到原點的距離
    gl_Position = MVP * vec4(aPos, 1.0);
}
"""

fragment_shader_source = """
#version 330 core
in float vDistance;
out vec4 FragColor;

uniform bool useUniformColor;  // 若為 true 表示使用 uColor（例如格線／軸線）
uniform vec4 uColor;           // 畫面統一顏色（格線、軸線的顏色）
uniform float maxDistance;     // 點雲色彩映射用的最大距離

vec3 jetColor(float t) {
    // 這個函式用來模擬 Jet Colormap 效果
    float r = clamp(min(4.0 * t - 1.5, -4.0 * t + 4.5), 0.0, 1.0);
    float g = clamp(min(4.0 * t - 0.5, -4.0 * t + 3.5), 0.0, 1.0);
    float b = clamp(min(4.0 * t + 0.5, -4.0 * t + 2.5), 0.0, 1.0);
    return vec3(r, g, b);
}

void main(){
    if(useUniformColor) {
        // 當繪製格線或軸線時，直接輸出傳入的統一顏色
        FragColor = uColor;
    } else {
        // 當繪製點雲時依據距離值計算對應顏色
        float d = clamp(vDistance / maxDistance, 0.0, 1.0);
        vec3 color = jetColor(d);
        FragColor = vec4(color, 1.0);
    }
}

"""

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

            num_points = struct.unpack('<I', data[:4])[0]
            expected_size = 4 + num_points * 3 * 4
            if len(data) != expected_size:
                continue

            floats = struct.unpack('<' + 'f' * (num_points * 3), data[4:])
            pts = np.array(floats, dtype=np.float64).reshape(-1, 3)

            # 使用每筆點的系統時間戳
            t_now = time.time()
            timestamps = np.full((pts.shape[0], 1), t_now, dtype=np.float64)

            new_points = np.hstack((pts, timestamps))  # [x, y, z, timestamp]

            with self.lock:
                self.ring_buffer.add_points(new_points)

# -------------------------------
# 點雲視覺化主類別
# -------------------------------
class PointCloudViewer:
    def __init__(self, width=1600, height=800):
        self.width = width
        self.height = height
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.last_cursor_pos = None
        self.zoom = 20.0
        self.retention_seconds = 5.0

        self.ring_buffer = PointRingBuffer(max_points=1920000)
        self.points_lock = threading.Lock()

        self.drag_threshold = 5.0  # 拖曳與點擊的像素容忍差
        self.mouse_moved = False   # 是否已被判斷為滑動
        self.selected_point = None  # 儲存被選中的點雲座標
        self.selecting_point = False  # 是否進入選點模式

        self.udp_receiver = UDPReceiver('0.0.0.0', 8080, self.ring_buffer, self.points_lock)
        self.udp_receiver.start()
        
        self.init_glfw()
        self.init_imgui()  # 必須在 shaders 與 buffers 之前初始化 imgui
        self.init_shaders()
        self.init_buffers()

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
        glEnable(GL_DEPTH_TEST)

    def init_imgui(self):
        imgui.create_context()
        self.impl = GlfwRenderer(self.window)

    def init_shaders(self):
        self.shader_program = create_shader_program(vertex_shader_source, fragment_shader_source)
        self.mvp_loc = glGetUniformLocation(self.shader_program, "MVP")
        self.color_loc = glGetUniformLocation(self.shader_program, "uColor")  # 用於 grid 和 axes
        self.max_distance = 5.0  # 初始顏色漸層最大距離
        self.max_distance_loc = glGetUniformLocation(self.shader_program, "maxDistance")


    def init_buffers(self):
        self.max_points = 1000000
        self.point_vao = glGenVertexArrays(1)
        self.point_vbo = glGenBuffers(1)
        glBindVertexArray(self.point_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.point_vbo)
        glBufferData(GL_ARRAY_BUFFER, self.max_points * 3 * 4, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glBindVertexArray(0)

        # Grid
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

        # Axes
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
                self.mouse_moved = False
            elif action == glfw.RELEASE:
                if not self.mouse_moved and self.selecting_point:
                    self.pick_point_at_cursor()
                    self.selecting_point = False  # 完成後自動退出選點模式
                self.last_cursor_pos = None


    def cursor_pos_callback(self, window, xpos, ypos):
        if self.last_cursor_pos is not None:
            last_x, last_y = self.last_cursor_pos
            dx = abs(xpos - last_x)
            dy = abs(ypos - last_y)
            if dx > self.drag_threshold or dy > self.drag_threshold:
                self.mouse_moved = True
            self.last_cursor_pos = (xpos, ypos)

    def handle_mouse_input(self):
        if self.last_cursor_pos is not None:
            xpos, ypos = glfw.get_cursor_pos(self.window)
            last_x, last_y = self.last_cursor_pos
            dx = xpos - last_x
            dy = ypos - last_y
            sensitivity = 0.005
            self.rotation_y += dx * sensitivity
            self.rotation_x += dy * sensitivity
            self.last_cursor_pos = (xpos, ypos)

    def handle_keyboard_input(self):
        rotation_speed = 0.05
        zoom_step = 0.3
        if glfw.get_key(self.window, glfw.KEY_A) == glfw.PRESS:
            self.rotation_y -= rotation_speed
        if glfw.get_key(self.window, glfw.KEY_D) == glfw.PRESS:
            self.rotation_y += rotation_speed
        if glfw.get_key(self.window, glfw.KEY_W) == glfw.PRESS:
            self.rotation_x -= rotation_speed
        if glfw.get_key(self.window, glfw.KEY_S) == glfw.PRESS:
            self.rotation_x += rotation_speed
        if glfw.get_key(self.window, glfw.KEY_Q) == glfw.PRESS:
            self.zoom = max(self.zoom - zoom_step, 0.1)
        if glfw.get_key(self.window, glfw.KEY_E) == glfw.PRESS:
            self.zoom += zoom_step

    def handle_scroll_input(self):
        # 利用 imgui.io 讀取滾輪輸入
        io = imgui.get_io()
        if io.mouse_wheel != 0.0:
            zoom_step = 1.0  # 可根據需要調整滾輪靈敏度
            self.zoom = max(self.zoom - io.mouse_wheel * zoom_step, 0.1)
            # 重置 mouse_wheel，避免累計效應
            io.mouse_wheel = 0.0

    def update(self):
        self.handle_keyboard_input()
        self.handle_mouse_input()
        self.handle_scroll_input()

    def render(self):
        glClearColor(0.2, 0.2, 0.2, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # 建立 view 矩陣 (依舊使用原來的參數)
        view = pyrr.matrix44.create_look_at(
            np.array([0.0, -self.zoom, 5.0], dtype=np.float32),
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32)
        )
        # 原有旋轉：pitch 為 X 軸旋轉、yaw 為 Z 軸旋轉
        pitch_matrix = pyrr.matrix44.create_from_x_rotation(self.rotation_x)
        yaw_matrix = pyrr.matrix44.create_from_z_rotation(self.rotation_y)
        # 1.先組合使用者控制的旋轉
        model = pyrr.matrix44.multiply(yaw_matrix, pitch_matrix)
        # 加入翻轉矩陣以修正由上向下且倒置的 LiDAR 資料
        # 2. 校正倒置: 沿 X 軸翻轉 180 度
        flip_transform = pyrr.matrix44.create_from_x_rotation(np.pi)
        # model = pyrr.matrix44.multiply(model, flip_transform)
        # 注意：這邊選擇右乘，即先對原始點雲做 flip，再套用使用者的旋轉，
        # 若需要使使用者旋轉不受翻轉影響，可改為左乘，但這取決於您的需求。


        # # 3. 平移: 將模型上移到 XY 平面以上
        # translation = pyrr.matrix44.create_from_translation(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        # model = pyrr.matrix44.multiply(translation, model)

        model = pyrr.matrix44.multiply(model, flip_transform)

        MVP = pyrr.matrix44.multiply(model, view)
        MVP = pyrr.matrix44.multiply(MVP, self.projection)

        # 儲存最後的 Model/View/Projection 矩陣，供選點使用
        self.last_model = model
        self.last_view = view
        self.last_projection = self.projection

        glUseProgram(self.shader_program)
        glUniformMatrix4fv(self.mvp_loc, 1, GL_FALSE, MVP)

        # 傳遞 model 與 view 矩陣給 Shader
        model_loc = glGetUniformLocation(self.shader_program, "model")
        view_loc = glGetUniformLocation(self.shader_program, "view")
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
        glUniformMatrix4fv(self.mvp_loc, 1, GL_FALSE, MVP)
        glUniform1f(self.max_distance_loc, self.max_distance)  # 設定點雲色帶映射最大距離

        # ------------------------------
        # 繪製點雲: 使用距離色彩映射 (useUniformColor = 0)
        glUniform1i(glGetUniformLocation(self.shader_program, "useUniformColor"), 0)
        with self.points_lock:
            pts_array = self.ring_buffer.get_recent_points(self.retention_seconds)
            total_pts = self.ring_buffer.size
            num_points = pts_array.shape[0]
        if num_points > 0:
            if num_points > self.max_points:
                pts_array = pts_array[:self.max_points]
                num_points = self.max_points
            pts_array = np.ascontiguousarray(pts_array, dtype=np.float32)
            glBindBuffer(GL_ARRAY_BUFFER, self.point_vbo)
            glBufferSubData(GL_ARRAY_BUFFER, 0, pts_array.nbytes, pts_array)
            glBindVertexArray(self.point_vao)
            glDrawArrays(GL_POINTS, 0, num_points)
            glBindVertexArray(0)
        if self.selected_point is not None:
            selected = np.array(self.selected_point, dtype=np.float32)

            glUseProgram(self.shader_program)
            glUniformMatrix4fv(self.mvp_loc, 1, GL_FALSE, MVP)
            glUniform1i(glGetUniformLocation(self.shader_program, "useUniformColor"), 1)
            glUniform4f(glGetUniformLocation(self.shader_program, "uColor"), 1.0, 0.0, 0.0, 1.0)  # 紅色

            glPointSize(10.0)  # 放大點
            glBindBuffer(GL_ARRAY_BUFFER, self.point_vbo)
            glBufferSubData(GL_ARRAY_BUFFER, 0, selected.nbytes, selected)
            glBindVertexArray(self.point_vao)
            glDrawArrays(GL_POINTS, 0, 1)
            glBindVertexArray(0)
            glPointSize(1.0)  # 還原

        # ------------------------------
        # 繪製格線: 使用固定顏色 (useUniformColor = 1)
        glUniform1i(glGetUniformLocation(self.shader_program, "useUniformColor"), 1)
        glBindVertexArray(self.grid_vao)
        glUniform4f(glGetUniformLocation(self.shader_program, "uColor"), 0.7, 0.7, 0.7, 1.0)
        glDrawArrays(GL_LINES, 0, self.grid_vertex_count)
        glBindVertexArray(0)

        # ------------------------------
        # 繪製軸線: 分別設定固定顏色 (useUniformColor = 1)
        glUniform1i(glGetUniformLocation(self.shader_program, "useUniformColor"), 1)
        glBindVertexArray(self.axes_vao)
        # X 軸: 紅色
        glUniform4f(glGetUniformLocation(self.shader_program, "uColor"), 1.0, 0.0, 0.0, 1.0)
        glDrawArrays(GL_LINES, 0, 2)
        # Y 軸: 綠色
        glUniform4f(glGetUniformLocation(self.shader_program, "uColor"), 0.0, 1.0, 0.0, 1.0)
        glDrawArrays(GL_LINES, 2, 2)
        # Z 軸: 藍色
        glUniform4f(glGetUniformLocation(self.shader_program, "uColor"), 0.0, 0.0, 1.0, 1.0)
        glDrawArrays(GL_LINES, 4, 2)
        glBindVertexArray(0)

        # ------------------------------
        # ImGui 控制面板
        imgui.new_frame()
        imgui.begin("Control Panel")
        _, self.retention_seconds = imgui.slider_float("Storage seconds", self.retention_seconds, 1.0, 30.0)
        _, self.max_distance = imgui.slider_float("Max distance (for color)", self.max_distance, 1.0, 20.0)
        
        if imgui.button("Clear Point Cloud"):
            with self.points_lock:
                self.ring_buffer.clear()

        if imgui.button("Save to .PLY"):
            self.save_points_to_ply()

        with self.points_lock:
            pts_array = self.ring_buffer.get_recent_points(self.retention_seconds)
            total_pts = self.ring_buffer.size
            num_points = pts_array.shape[0]

        if self.selected_point is not None:
            imgui.text("Selected Point:")
            imgui.text(f"X: {self.selected_point[0]:.2f}")
            imgui.text(f"Y: {self.selected_point[1]:.2f}")
            imgui.text(f"Z: {self.selected_point[2]:.2f}")
        else:
            imgui.text("Selected Point: None")

        imgui.text(f"Current points: {num_points}")
        imgui.text(f"Total stored: {total_pts} points")
        imgui.text(f"Displayed: {num_points} (within {self.retention_seconds:.1f}s)")
        if imgui.button("Select a Point"):
            self.selecting_point = True
            print("請在畫面中點選要選擇的點")
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

    def save_points_to_ply(self):
        with self.points_lock:
            points = self.ring_buffer.get_recent_points(self.retention_seconds)
        if points.size == 0:
            print("沒有可儲存的點雲資料")
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"pointcloud_{timestamp}.ply"

        with open(filename, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            for p in points:
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")

        print(f"已儲存 {len(points)} 筆點雲資料至 {filename}")

    def pick_point_at_cursor(self):
        xpos, ypos = glfw.get_cursor_pos(self.window)
        mouse_screen = np.array([xpos, ypos], dtype=np.float32)

        with self.points_lock:
            points = self.ring_buffer.get_recent_points(self.retention_seconds)
        if points.shape[0] == 0:
            self.selected_point = None
            print("[選取失敗] 沒有點雲")
            return

        # 將所有點轉換成 clip space → NDC → screen space
        pts_homo = np.hstack((points, np.ones((points.shape[0], 1), dtype=np.float32)))  # Nx4

        MVP = pyrr.matrix44.multiply(
            pyrr.matrix44.multiply(self.last_model, self.last_view),
            self.last_projection
        )

        clip_coords = pts_homo @ MVP.T  # Nx4
        ndc = clip_coords[:, :3] / clip_coords[:, 3:4]  # Nx3 → NDC 空間
        screen_coords = np.empty((points.shape[0], 2), dtype=np.float32)
        screen_coords[:, 0] = (ndc[:, 0] * 0.5 + 0.5) * self.width
        screen_coords[:, 1] = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * self.height

        # 計算滑鼠與每個投影點的距離
        dists_2d = np.linalg.norm(screen_coords - mouse_screen, axis=1)
        min_idx = np.argmin(dists_2d)

        # 設定容差（像素），避免滑鼠在空白區誤選
        pixel_tolerance = 30.0
        if dists_2d[min_idx] < pixel_tolerance:
            self.selected_point = points[min_idx]
            print(f"[選取成功] 螢幕距離 {dists_2d[min_idx]:.1f}px → 點座標：{self.selected_point}")
        else:
            self.selected_point = None
            print("[選取失敗] 沒有點靠近滑鼠")




# -------------------------------
# 程式進入點
# -------------------------------
if __name__ == '__main__':
    viewer = PointCloudViewer()
    viewer.run()

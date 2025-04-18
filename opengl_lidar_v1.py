import glfw                       # 用於建立與管理視窗及處理輸入
import threading
from OpenGL.GL import *           # OpenGL API，用於實際渲染圖形
import numpy as np                # 高效數值計算與陣列操作
import pyrr                       # 提供矩陣、向量及旋轉計算（例如旋轉矩陣與透視投影）
import socket                     # 用於網路通訊，這裡採用 UDP 協定接收點雲資料
import struct                     # 用於封包中數據的編解碼（例如解析 float 與 int）
import threading                  # 使用多執行緒實現非同步資料接收
import time                       # 計時、取得系統時間（用來建立時間戳）
import imgui                      # 即時 GUI 套件，提供控制面板
from imgui.integrations.glfw import GlfwRenderer  # 將 imgui 與 glfw 整合，方便 GUI 輸入與繪製
from scipy.spatial import cKDTree # KD-Tree (目前範例中未直接應用，但可供後續擴充)

# =============================================================================
# PointRingBuffer 類別：環狀緩衝區
# -----------------------------------------------------------------------------
# 用於固定大小內高效儲存點雲資料與時間戳，當空間滿時會覆蓋最舊資料
# =============================================================================
class PointRingBuffer:
    def __init__(self, max_points):
        """
        初始化環狀緩衝區
        
        參數:
            max_points (int): 可儲存的最大點數
        """
        self.max_points = max_points
        # 建立固定大小的 NumPy 陣列，每一筆資料包含 (x, y, z, timestamp)
        self.buffer = np.zeros((max_points, 4), dtype=np.float64)
        self.index = 0     # 下次插入資料的位置
        self.size = 0      # 緩衝區目前有效的資料個數

    def add_points(self, new_points: np.ndarray):
        """
        將新點加入環狀緩衝區，若超過容量則覆蓋最舊資料
        
        參數:
            new_points (np.ndarray): 必須為 Nx4 陣列，包含 x, y, z 與 timestamp
        """
        if new_points is None or new_points.size == 0:
            return

        if new_points.shape[1] != 4:
            raise ValueError("新加入的點必須為 Nx4 陣列（x, y, z, timestamp）")
        n = new_points.shape[0]
        end_index = self.index + n

        if end_index <= self.max_points:
            # 未超出緩衝區大小，直接放置
            self.buffer[self.index:end_index] = new_points
        else:
            # 超出情形，先放入尾端，剩餘部分從開頭覆蓋
            overflow = end_index - self.max_points
            self.buffer[self.index:] = new_points[:n - overflow]
            self.buffer[:overflow] = new_points[n - overflow:]
        self.index = (self.index + n) % self.max_points
        self.size = min(self.size + n, self.max_points)

    def get_recent_points(self, retention_time):
        """
        取得保留時間（秒）內的最新點雲資料，只保留 x, y, z 座標，
        利用向量化的 np.roll 取代 np.concatenate 來獲得連續的資料區塊。
        """
        if self.size == 0:
            return np.empty((0, 3), dtype=np.float32)
            
        now = time.time()
        cutoff = now - retention_time

        # 當資料尚未滿時，直接取前 self.size 筆；滿時用 np.roll 使資料連續
        if self.size < self.max_points:
            data = self.buffer[:self.size]
        else:
            # np.roll 是向量化的，不會在 Python 迴圈中逐筆操作
            data = np.roll(self.buffer, -self.index, axis=0)

        # data 裡面的 timestamp 應該是遞增的，可以使用 np.searchsorted 快速定位
        idx = np.searchsorted(data[:, 3], cutoff, side='left')
        valid = data[idx:, :3]
        
        # 如果原本的資料型別不是 float32，再進行轉換也使用向量化操作
        return valid.astype(np.float32)



    def clear(self):
        """ 清除緩衝區中的所有點雲資料 """
        self.buffer[:] = 0
        self.index = 0
        self.size = 0

# =============================================================================
# Shader 工具函式：負責編譯與連結 shader 程式
# =============================================================================
def compile_shader(shader_type, source):
    """
    編譯給定來源碼的 shader
    
    參數:
        shader_type: GL_VERTEX_SHADER 或 GL_FRAGMENT_SHADER
        source (str): shader 原始碼
    回傳:
        編譯完成後的 shader ID
    """
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
        error = glGetShaderInfoLog(shader).decode()
        raise RuntimeError("Shader compile error: " + error)
    return shader

def create_shader_program(vertex_src, fragment_src):
    """
    建立 shader 程式（Program），由 vertex 與 fragment shader 組成
    
    參數:
        vertex_src (str): vertex shader 原始碼
        fragment_src (str): fragment shader 原始碼
    回傳:
        已連結的 shader 程式 ID
    """
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
# Vertex Shader：將點雲頂點從物件空間轉換到 clip space，同時計算到原點的距離
# -----------------------------------------------------------------------------
vertex_shader_source = """
#version 330 core
layout(location = 0) in vec3 aPos;   // 輸入頂點座標
uniform mat4 MVP;                  // Model-View-Projection 矩陣
out float vDistance;               // 傳遞頂點到原點的距離給 fragment shader
void main(){
    vDistance = length(aPos);      // 計算頂點到原點的距離
    gl_Position = MVP * vec4(aPos, 1.0);
}
"""

# -----------------------------------------------------------------------------
# Fragment Shader：根據傳入的距離資訊使用 Jet colormap 計算顏色，
# 或在指定情況下使用統一顏色（例如格線或軸線）
# -----------------------------------------------------------------------------
fragment_shader_source = """
#version 330 core
in float vDistance;                // 從 vertex shader 傳入的距離資訊
out vec4 FragColor;                // 輸出像素最終顏色
uniform bool useUniformColor;      // 是否使用統一顏色
uniform vec4 uColor;               // 固定的統一顏色
uniform float maxDistance;         // 用於顏色映射的最大距離參數
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
# UDPReceiver 類別：持續接收 UDP 封包並解析出點雲資料，
# 然後利用多執行緒安全機制將資料加入環狀緩衝區中
# =============================================================================
class UDPReceiver(threading.Thread):
    def __init__(self, host, port, ring_buffer, lock):
        """
        初始化 UDP 接收器
        
        參數:
          host (str): 監聽的主機位址（例如 '0.0.0.0' 監聽所有介面）
          port (int): UDP 埠號
          ring_buffer (PointRingBuffer): 儲存點雲資料的環狀緩衝區
          lock (threading.Lock): 用於多執行緒存取緩衝區的鎖
        """
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.ring_buffer = ring_buffer
        self.lock = lock

    def run(self):
        """
        持續執行迴圈：接收 UDP 封包、解析點雲資料，
        並加上當前時間戳後存入環狀緩衝區。
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.host, self.port))
        print(f"UDP 伺服器正在監聽 {self.host}:{self.port}")
        while True:
            data, addr = sock.recvfrom(65535)
            if not data or len(data) < 4:
                continue
            # 封包前 4 個 byte 為點數（unsigned int）
            num_points = struct.unpack('<I', data[:4])[0]
            expected_size = 4 + num_points * 3 * 4
            if len(data) != expected_size:
                continue
            # 解碼封包內的浮點數後重組為 (N, 3) 陣列
            floats = struct.unpack('<' + 'f' * (num_points * 3), data[4:])
            pts = np.array(floats, dtype=np.float64).reshape(-1, 3)
            # 取得當前系統時間作為每筆資料的時間戳
            t_now = time.time()
            timestamps = np.full((pts.shape[0], 1), t_now, dtype=np.float64)
            new_points = np.hstack((pts, timestamps))
            # 利用鎖確保在多執行緒環境下安全更新緩衝區
            with self.lock:
                self.ring_buffer.add_points(new_points)

# =============================================================================
# PointCloudViewer 類別：主要負責點雲資料的視覺化與使用者互動
# =============================================================================
class PointCloudViewer:
    def __init__(self, width=1600, height=800):
        """
        初始化視窗與各項控制元件，並設定各種初始參數
        
        新增參數:
          pan_offset: 三維向量，用於實現視角平移（攝影機移動）
            - 原本的 WASD/QE 控制值將會交換：
                WASD：除了 A/D 水平移動以外，W/S 將用於垂直平移 (Y 軸)
                      Q/E 將用於前後移動 (Z 軸)
        """
        self.width = width
        self.height = height
        # 旋轉參數，保留滑鼠拖曳旋轉與方向鍵旋轉（Arrow keys）
        self.rotation_x = 0.0  
        self.rotation_y = 0.0  
        self.last_cursor_pos = None
        self.zoom = 20.0              # 縮放參數（視距遠近），由滾輪控制
        self.retention_seconds = 5.0  # 顯示點雲資料的保留時間（秒）
        # 初始化平移偏移量（pan_offset），作為攝影機位置的位移
        self.pan_offset = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # 初始化環狀緩衝區與鎖，儲存來自 UDP 的點雲資料
        self.ring_buffer = PointRingBuffer(max_points=10000000)
        self.points_lock = threading.Lock()

        # 啟動 UDP 接收線程，持續接收並儲存點雲資料
        self.udp_receiver = UDPReceiver('0.0.0.0', 8080, self.ring_buffer, self.points_lock)
        self.udp_receiver.start()

        # 初始化 GLFW（視窗與 OpenGL 環境）、ImGui（GUI 控制面板）、Shader 與緩衝區
        self.init_glfw()
        self.init_imgui()
        self.init_shaders()
        self.init_buffers()

        # 啟動 TCP 傳送線程，連線至接收端（例如本機 127.0.0.1:9000）
        #self.tcp_sender_thread = threading.Thread(target=self.tcp_sender, args=("127.0.0.1", 9000), daemon=True)
        self.tcp_sender_thread = threading.Thread(target=self.tcp_sender, args=("192.168.50.231", 9000), daemon=True)
        self.tcp_sender_thread.start()


    def tcp_sender(self, host, port):
        """
        TCP 傳送線程：根據目前 retention_seconds 設定，定期取得點雲資料，
        並將其以二進制方式傳送到指定的接收端。
        """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((host, port))
            print(f"TCP Sender: 連線到 {host}:{port}")
        except Exception as e:
            print(f"TCP Sender 無法連線: {e}")
            return
        
        while True:
            # 取得指定 retention_seconds 內的點雲資料
            with self.points_lock:
                points = self.ring_buffer.get_recent_points(self.retention_seconds)
            # 將資料轉換成 float32 並轉為二進制字串
            points_bin = points.astype(np.float32).tobytes()
            # 傳送資料前，先傳送資料長度（4 bytes 的 unsigned int）
            header = struct.pack('<I', len(points_bin))
            try:
                sock.sendall(header + points_bin)
            except Exception as e:
                print(f"TCP Sender 傳送錯誤: {e}")
                break
            # 每隔 100ms 傳一次，你可根據需要調整頻率
            time.sleep(self.retention_seconds)

    def restart_tcp_connection(self, host, port):
        """
        檢查並重啟 TCP 傳送線程
        """
        # 若傳送線程不存在或已不在運作，重啟線程
        if not hasattr(self, 'tcp_sender_thread') or not self.tcp_sender_thread.is_alive():
            print("TCP 傳送線程已停止，正在重啟...")
            self.tcp_sender_thread = threading.Thread(target=self.tcp_sender, args=(host, port), daemon=True)
            self.tcp_sender_thread.start()
        else:
            print("TCP 傳送線程仍在運行。")



    # -------------------------------------------------------------------------
    # GLFW 與 OpenGL 環境初始化
    # -------------------------------------------------------------------------
    def init_glfw(self):
        """
        初始化 GLFW，建立 OpenGL 視窗以及設定滑鼠 callback
        """
        if not glfw.init():
            raise Exception("GLFW 初始化失敗")
        self.window = glfw.create_window(self.width, self.height, "UDP Point Cloud Viewer", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("無法建立視窗")
        glfw.make_context_current(self.window)
        # 設定滑鼠按鍵與滑鼠移動 callback（移除點選相關功能）
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_cursor_pos_callback(self.window, self.cursor_pos_callback)
        glEnable(GL_DEPTH_TEST)

    def init_imgui(self):
        """
        初始化 ImGui，建立即時 GUI 控制面板
        """
        imgui.create_context()
        self.impl = GlfwRenderer(self.window)

    def init_shaders(self):
        """
        建立 shader 程式（由 vertex 與 fragment shader 組成），
        並取得 MVP 與顏色映射參數的位置
        """
        self.shader_program = create_shader_program(vertex_shader_source, fragment_shader_source)
        self.mvp_loc = glGetUniformLocation(self.shader_program, "MVP")
        self.max_distance_loc = glGetUniformLocation(self.shader_program, "maxDistance")
        self.max_distance = 5.0

    def init_buffers(self):
        """
        初始化 OpenGL 緩衝區與頂點陣列物件 (VAO)，包含：
          - 點雲資料的 VAO 與 VBO
          - 格線資料（作為背景輔助參考）
          - XYZ 軸線（用於標示空間方向）
        """
        # 初始化點雲資料的 VAO 與 VBO
        self.max_points = 19200000  # 每次渲染最多處理的點數
        self.point_vao = glGenVertexArrays(1)
        self.point_vbo = glGenBuffers(1)
        glBindVertexArray(self.point_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.point_vbo)
        glBufferData(GL_ARRAY_BUFFER, self.max_points * 3 * 4, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glBindVertexArray(0)

        # 建立格線：以水平與垂直線條構成背景網格
        grid_size = 10
        grid_lines = []
        for i in range(-grid_size, grid_size + 1):
            # 水平線：從 (-grid_size, i) 到 (grid_size, i)
            grid_lines.extend([-grid_size, i, 0.0, grid_size, i, 0.0])
            # 垂直線：從 (i, -grid_size) 到 (i, grid_size)
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

        # 建立 XYZ 軸線：分別以紅、綠、藍三色表示 X、Y、Z 軸
        axes_vertices = np.array([
            0, 0, 0, 1, 0, 0,  # X 軸：紅色
            0, 0, 0, 0, 1, 0,  # Y 軸：綠色
            0, 0, 0, 0, 0, 1   # Z 軸：藍色
        ], dtype=np.float32)
        self.axes_vao = glGenVertexArrays(1)
        self.axes_vbo = glGenBuffers(1)
        glBindVertexArray(self.axes_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.axes_vbo)
        glBufferData(GL_ARRAY_BUFFER, axes_vertices.nbytes, axes_vertices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glBindVertexArray(0)

        # 建立透視投影矩陣：設定視場角、寬高比以及近遠裁切平面
        self.projection = pyrr.matrix44.create_perspective_projection_matrix(
            45, self.width / self.height, 0.1, 100.0
        )

    # -------------------------------------------------------------------------
    # 輸入事件處理：滑鼠 callback（移除點選相關邏輯）
    # -------------------------------------------------------------------------
    def mouse_button_callback(self, window, button, action, mods):
        """
        處理滑鼠按鍵事件：
          - 按下左鍵時記錄位置以便旋轉
        """
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                self.last_cursor_pos = glfw.get_cursor_pos(window)
            elif action == glfw.RELEASE:
                self.last_cursor_pos = None

    def cursor_pos_callback(self, window, xpos, ypos):
        """
        處理滑鼠移動事件：更新最近的滑鼠位置
        """
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

            # 限制上下角度：避免轉太過頭造成「上下顛倒」
            max_pitch = np.pi / 2 - 0.01
            self.rotation_x = np.clip(self.rotation_x, -max_pitch, max_pitch)

            self.last_cursor_pos = (xpos, ypos)

    # -------------------------------------------------------------------------
    # 鍵盤輸入處理：
    # 使用方向鍵控制旋轉，WASD 與 Q/E 控制平移（視角位移）
    # -------------------------------------------------------------------------
    def handle_keyboard_input(self):
        rotation_speed = 0.01
        pan_step = 0.1

        # 方向鍵控制旋轉（原樣保留）
        if glfw.get_key(self.window, glfw.KEY_LEFT) == glfw.PRESS:
            self.rotation_y -= rotation_speed
        if glfw.get_key(self.window, glfw.KEY_RIGHT) == glfw.PRESS:
            self.rotation_y += rotation_speed
        if glfw.get_key(self.window, glfw.KEY_UP) == glfw.PRESS:
            self.rotation_x -= rotation_speed
        if glfw.get_key(self.window, glfw.KEY_DOWN) == glfw.PRESS:
            self.rotation_x += rotation_speed

        # 計算旋轉後的前向、右向、上向方向向量
        yaw_matrix = pyrr.matrix33.create_from_z_rotation(self.rotation_y)
        pitch_matrix = pyrr.matrix33.create_from_x_rotation(self.rotation_x)
        rotation_matrix = pyrr.matrix33.multiply(pitch_matrix, yaw_matrix)

        forward = pyrr.vector3.normalize(rotation_matrix @ np.array([0, 0, -1], dtype=np.float32))  # 向前
        right   = pyrr.vector3.normalize(rotation_matrix @ np.array([1, 0, 0], dtype=np.float32))   # 向右
        up      = pyrr.vector3.normalize(np.array([0, 1, 0], dtype=np.float32))                     # 上方維持世界座標 Y

        # WASD/QE 鍵控制：根據旋轉方向平移
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

        # PgUp / PgDn 控制 zoom
        zoom_step = 0.1
        if glfw.get_key(self.window, glfw.KEY_PAGE_UP) == glfw.PRESS:
            self.zoom = max(0.1, self.zoom - zoom_step)
        if glfw.get_key(self.window, glfw.KEY_PAGE_DOWN) == glfw.PRESS:
            self.zoom += zoom_step


    def handle_scroll_input(self):
        """
        透過 ImGui 滾輪輸入調整 zoom（視距遠近）
        """
        io = imgui.get_io()
        if io.mouse_wheel != 0.0:
            self.zoom = max(self.zoom - io.mouse_wheel * 1.0, 0.1)
            io.mouse_wheel = 0.0

    def update(self):
        """
        更新所有輸入：
          - 鍵盤輸入（旋轉與平移）
          - 滑鼠拖曳產生旋轉
          - 滾輪產生 zoom 調整
        """
        self.handle_keyboard_input()
        self.handle_mouse_input()
        self.handle_scroll_input()

    # -------------------------------------------------------------------------
    # 渲染函式：
    # 生成 view 與 MVP 矩陣，並根據這些矩陣渲染點雲、格線與軸線，
    # 同時利用 ImGui 顯示控制面板資訊
    # -------------------------------------------------------------------------
    def render(self):
        glClearColor(0.2, 0.2, 0.2, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # 建立 view 矩陣
        base_eye = np.array([0.0, -self.zoom, 5.0], dtype=np.float32)
        base_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        eye = base_eye + self.pan_offset
        target = base_target + self.pan_offset
        view = pyrr.matrix44.create_look_at(eye, target, np.array([0.0, 1.0, 0.0], dtype=np.float32))

        # 建立模型矩陣：利用旋轉參數產生旋轉效果，並做翻轉調整
        pitch = pyrr.matrix44.create_from_x_rotation(self.rotation_x)
        yaw = pyrr.matrix44.create_from_z_rotation(self.rotation_y)
        model = pyrr.matrix44.multiply(yaw, pitch)
        #flip = pyrr.matrix44.create_from_x_rotation(np.pi)
        #model = pyrr.matrix44.multiply(model, flip)

        # 計算 MVP 矩陣 = model * view * projection
        MVP = pyrr.matrix44.multiply(model, view)
        MVP = pyrr.matrix44.multiply(MVP, self.projection)
        # 儲存部分矩陣供未來使用
        self.last_model, self.last_view, self.last_projection = model, view, self.projection

        # 使用 shader 程式並將 MVP 及最大距離傳遞給 shader
        glUseProgram(self.shader_program)
        glUniformMatrix4fv(self.mvp_loc, 1, GL_FALSE, MVP)
        glUniform1f(self.max_distance_loc, self.max_distance)

        # ---------------------------------------------------------------------
        # 繪製點雲資料：
        # 從環狀緩衝區取得點雲，將資料傳送到 GPU 並以 Jet Colormap 渲染
        # ---------------------------------------------------------------------
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

        # ---------------------------------------------------------------------
        # 繪製格線（背景輔助參考）
        # ---------------------------------------------------------------------
        glUniform1i(glGetUniformLocation(self.shader_program, "useUniformColor"), 1)
        glBindVertexArray(self.grid_vao)
        glUniform4f(glGetUniformLocation(self.shader_program, "uColor"), 0.7, 0.7, 0.7, 1.0)
        glDrawArrays(GL_LINES, 0, self.grid_vertex_count)
        glBindVertexArray(0)

        # ---------------------------------------------------------------------
        # 繪製 XYZ 軸線：分別以紅、綠、藍表示 X、Y、Z 軸
        # ---------------------------------------------------------------------
        glBindVertexArray(self.axes_vao)
        glUniform4f(glGetUniformLocation(self.shader_program, "uColor"), 1.0, 0.0, 0.0, 1.0)
        glDrawArrays(GL_LINES, 0, 2)
        glUniform4f(glGetUniformLocation(self.shader_program, "uColor"), 0.0, 1.0, 0.0, 1.0)
        glDrawArrays(GL_LINES, 2, 2)
        glUniform4f(glGetUniformLocation(self.shader_program, "uColor"), 0.0, 0.0, 1.0, 1.0)
        glDrawArrays(GL_LINES, 4, 2)
        glBindVertexArray(0)

        # ---------------------------------------------------------------------
        # ImGui 控制面板：
        # 顯示參數設定（保留時間、最大距離）、點數資訊，
        # 並提供清除、存檔與視角重設功能
        # ---------------------------------------------------------------------
        imgui.new_frame()
        imgui.begin("Control Panel")
        _, self.retention_seconds = imgui.slider_float("Storage seconds", self.retention_seconds, 1.0, 30.0)
        _, self.max_distance = imgui.slider_float("Max distance", self.max_distance, 1.0, 20.0)
        if imgui.button("Clear Point Cloud"):
            with self.points_lock:
                self.ring_buffer.clear()
        if imgui.button("Save to .PLY"):
            self.save_points_to_ply()
        # 新增按鈕：Reset View 重設旋轉、平移與 zoom

        if imgui.button("Restart TCP Connection"):
            #self.restart_tcp_connection("127.0.0.1", 9000)
            self.restart_tcp_connection("192.168.50.231", 9000)
        if imgui.button("Reset View"):
            self.rotation_x = 0.0
            self.rotation_y = 0.0
            self.pan_offset = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self.zoom = 20.0
        imgui.text(f"Current points: {pts_array.shape[0]}")
        imgui.text(f"Total stored: {total_pts}")
        imgui.end()
        imgui.render()
        self.impl.render(imgui.get_draw_data())
        glfw.swap_buffers(self.window)

    def run(self):
        """
        主迴圈：持續處理輸入與更新畫面，
        直到使用者關閉視窗後執行資源清理
        """
        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            self.impl.process_inputs()
            self.update()
            self.render()
        self.cleanup()

    def cleanup(self):
        """
        清理資源，關閉 ImGui 並終止 GLFW
        """
        self.impl.shutdown()
        glfw.terminate()

    def save_points_to_ply_binary_task(self,points: np.ndarray, filename: str, header: str):
            """
            後台任務：將點雲資料以二進制格式寫入 PLY 檔案

            參數:
            points (np.ndarray): 要儲存的點陣列 (Nx3)
            filename (str): 儲存檔案名稱
            header (str): PLY 檔案的 header (ASCII)
            """
            try:
                with open(filename, 'wb') as f:
                    # 先寫入 header，轉為 UTF-8 編碼
                    f.write(header.encode('utf-8'))
                    # 將點資料轉成 float32 並直接以二進制寫入
                    points.astype(np.float32).tofile(f)
                print(f"已儲存 {len(points)} 筆點雲（二進制格式），儲存至 {filename}")
            except Exception as e:
                print(f"儲存點雲資料時發生錯誤: {e}")

    def save_points_to_ply(self):
        """
        將保留時間內的點雲資料儲存至 PLY 檔案，檔名根據當前日期與時間自動命名，
        並以非同步二進制方式寫入檔案。
        """
        with self.points_lock:
            points = self.ring_buffer.get_recent_points(self.retention_seconds)
        if points.size == 0:
            print("沒有可儲存的點雲資料")
            return
        filename = f"pointcloud_{time.strftime('%Y%m%d_%H%M%S')}_binary.ply"
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {len(points)}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "end_header\n"
        )
        # 利用背景執行緒執行檔案寫入，避免阻塞主線程
        threading.Thread(target=self.save_points_to_ply_binary_task,
                 args=(points, filename, header),
                 daemon=True).start()

    
    

        

# =============================================================================
# 程式進入點：建立並運行點雲視覺化應用
# =============================================================================
if __name__ == '__main__':
    viewer = PointCloudViewer()
    viewer.run()

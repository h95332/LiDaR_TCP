import cv2
import os
from datetime import datetime

# 設定儲存畫面的路徑
save_path = 'right_captured_frames'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 使用第一個攝像頭，並指定使用 DirectShow
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("無法開啟攝像頭")
    exit()
    
# 設定攝像頭解析度
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_ZOOM, 100)
cap.set(cv2.CAP_PROP_EXPOSURE,-6)
cap.set(cv2.CAP_PROP_GAIN, 3)
cap.set(cv2.CAP_PROP_SETTINGS, 1)

# 創建一個窗口
cv2.namedWindow('WebCam')

# 棋盘格配置
checkerboard_size = (5, 8)  # 5x8棋盤格

# 設定縮放比例
scale_percent = 50  # 縮放比例，50%表示影像縮小一半

while True:
    # 讀取一幀畫面
    ret, img = cap.read()
    if not ret:
        print("無法讀取畫面")
        break

    # 複製原始影像
    src = img.copy()

    # 轉換為灰度影像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 找到棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

    if ret:
        # 細化角點位置
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # 在每個角點上顯示其序號
        for i, corner in enumerate(corners2):
            x, y = corner.ravel()
            cv2.circle(src, (int(x), int(y)), 5, (255, 0, 0), -1)
            cv2.putText(src, str(i + 1), (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # 縮放影像
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    resized_src = cv2.resize(src, dim, interpolation=cv2.INTER_AREA)

    # 顯示影像
    cv2.imshow('WebCam', resized_img)
    if ret:
        cv2.imshow('Chessboard with Numbers', resized_src)

    # 等待按鍵指令
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        # 獲取當前時間
        now = datetime.now()
        # 格式化時間字符串
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        # 儲存畫面
        frame_name = os.path.join(save_path, f"frame_{timestamp}.jpg")
        cv2.imwrite(frame_name, img)
        print(f"儲存畫面到 {frame_name}")

    elif key == ord('q'):
        # 退出循環
        break

# 釋放攝像頭並關閉所有OpenCV視窗
cap.release()
cv2.destroyAllWindows()

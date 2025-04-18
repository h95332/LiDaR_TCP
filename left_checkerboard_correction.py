import cv2
import numpy as np
import glob

# mtx = [[387.27754435,   0,         313.61402983],
#         [  0,         387.74992969, 190.1021534 ],
#         [  0,           0,           1        ]]
# dist = [[ 0.10358343, -0.24711189, -0.00081118, -0.00054372,  0.11443435]]
# mtx = np.array(mtx)
# dist = np.array(dist)

# 棋盤格設定
checkerboard_size = (5, 8)  # 棋盤格角點的行和列數
square_size = 2.75  # 棋盤格中每個方塊的大小（單位不重要，保持一致即可）
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 準備物體點和圖像點的列表
objpoints = []  # 真實世界中的3D點
imgpoints = []  # 圖像中的2D點

# 準備物體點（0,0,0），(1,0,0)，(2,0,0) ... 如此類推
objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2) * square_size

# 讀取用於校準的圖像
images = glob.glob('right_captured_frames/*.jpg')
if not images:
    print("未找到任何圖像")
    exit()

# 讀取所有校準圖像並檢測棋盤格角點
for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"無法讀取圖像 {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 找到棋盤格角點
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # 畫出角點並顯示
        cv2.drawChessboardCorners(img, checkerboard_size, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)
    else:
        print(f"未能找到棋盤格角點 {fname}")

cv2.destroyAllWindows()

# 校準相機
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 打印校準結果
print("相機內部參數矩陣：")
print(mtx)
print("畸變係數：")
print(dist)

# 校正畸變
for fname in images:
    img = cv2.imread(fname)
    h, w = img.shape[:2]
    
    # 獲取優化過的相機矩陣
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    
    # 使用新的相機矩陣校正畸變
    dst = cv2.undistort(img, mtx, dist, None, new_camera_mtx)

    # 裁剪圖像
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    # 顯示校正結果
    cv2.imshow('calibresult', dst)
    cv2.waitKey(500)

cv2.destroyAllWindows()

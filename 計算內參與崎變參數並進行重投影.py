import cv2
import numpy as np
import glob

# 棋盘格设置
checkerboard_size = (5, 8)  # 棋盘格角点的行和列数
square_size = 2.75  # 棋盘格中每个方块的大小（单位不重要，保持一致即可）
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 准备物体点和图像点的列表
objpoints = []  # 真实世界中的3D点
imgpoints = []  # 图像中的2D点

# 准备物体点（0,0,0），(1,0,0)，(2,0,0) ... 如此类推
objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2) * square_size

# 读取用于校准的图像
# images = glob.glob('D:/Code/Paper_Code/Checkerboard_correction/left_captured_frames/*.jpg')
images = glob.glob('right_captured_frames/*.jpg')
if not images:
    print("未找到任何图像")
    exit()

# 读取所有校准图像并检测棋盘格角点
for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"无法读取图像 {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 找到棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # 画出角点并显示
        cv2.drawChessboardCorners(img, checkerboard_size, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)
    else:
        print(f"未能找到棋盘格角点 {fname}")

cv2.destroyAllWindows()

# 校准相机
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 打印校准结果
print("相机内参矩阵：")
print(mtx)
print("畸变系数：")
print(dist)

# 计算重投影误差
total_error = 0
for i in range(len(objpoints)):
    # 将3D点投影到图像平面
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    # 计算每个点的误差
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    total_error += error

mean_error = total_error / len(objpoints)
print(f"平均重投影误差： {mean_error}")

# 校正畸变
for fname in images:
    img = cv2.imread(fname)
    h, w = img.shape[:2]
    
    # 获取优化过的相机矩阵
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    
    # 使用新的相机矩阵校正畸变
    dst = cv2.undistort(img, mtx, dist, None, new_camera_mtx)

    # 裁剪图像
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    # 显示校正结果
    cv2.imshow('calibresult', dst)
    cv2.waitKey(500)

cv2.destroyAllWindows()

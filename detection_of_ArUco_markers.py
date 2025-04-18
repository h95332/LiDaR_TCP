import cv2
import cv2.aruco as aruco
import numpy as np

# 單一攝像頭的內參矩陣和畸變係數
camera_matrix = np.array([[387.27754435, 0, 313.61402983],
                          [0, 387.74992969, 190.1021534],
                          [0, 0, 1]], dtype=float)
dist_coeffs = np.array((0.10358343, -0.24711189, -0.00081118, -0.00054372, 0.11443435))

def detect_aruco_single_camera(camera_id=1, dictionary=aruco.DICT_4X4_250):
    """
    使用單一攝像頭偵測 ArUco 標記並繪製 3D 坐標軸。
    """
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("無法打開攝像頭")
        return

    aruco_dict = aruco.getPredefinedDictionary(dictionary)
    parameters = aruco.DetectorParameters()

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
            # 假設已經取得 rvecs[0] 與 tvecs[0]
            rvec = rvecs[0][0]  # 取出第 0 個標記的旋轉向量
            tvec = tvecs[0][0]  # 取出第 0 個標記的平移向量

            # 將旋轉向量轉換為旋轉矩陣
            rotation_matrix, _ = cv2.Rodrigues(rvec)

            # 建立 4x4 齊次變換矩陣
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rotation_matrix
            transform_matrix[:3, 3] = tvec
            print(transform_matrix)
            for rvec, tvec in zip(rvecs, tvecs):
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

        cv2.imshow('Camera - ArUco Markers with 3D Axis', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 呼叫函數，假設攝像頭 ID 為 1
detect_aruco_single_camera(1)

import cv2
import cv2.aruco as aruco
import numpy as np

def generate_aruco_marker(marker_id, marker_size=718, dictionary=aruco.DICT_4X4_50):      # 5cm的aruco Pixel為188.98, 4cm Pixel為151.18
    # 获取指定的字典
    aruco_dict = aruco.getPredefinedDictionary(dictionary)
    
    # 生成标记图像
    img = aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
    
    # 保存标记图像
    cv2.imwrite(f'aruco_marker_{marker_id}.png', img)
    print(f"已保存 ArUco 标记 ID {marker_id} 为 aruco_marker_{marker_id}.png")

# 生成 ID 为 0 到 4 的 ArUco 标记
for marker_id in range(2):
    generate_aruco_marker(marker_id)







# import cv2
# import cv2.aruco as aruco
# import numpy as np

# def generate_aruco_marker_with_trim_line(marker_id, marker_size=94, dictionary=aruco.DICT_6X6_50, trim_line_offset_cm=1):      # 5cm的aruco Pixel為188.98, 4cm Pixel為151.18, 16cm Pixel為604.72, 2.5cm Pixel為94
#     # 1公分大约等于37.8像素，假设DPI为96
#     trim_line_offset_px = int(37.8 * trim_line_offset_cm)

#     # 获取指定的字典
#     aruco_dict = aruco.getPredefinedDictionary(dictionary)
    
#     # 生成标记图像
#     img = aruco.generateImageMarker(aruco_dict, marker_id, marker_size)

#     # 在图像四周添加较大的白色边框，以便绘制修剪线
#     img_with_white_border = cv2.copyMakeBorder(img, 
#                                                trim_line_offset_px + 30, trim_line_offset_px + 30, 
#                                                trim_line_offset_px + 30, trim_line_offset_px + 30, 
#                                                cv2.BORDER_CONSTANT, 
#                                                value=[255, 255, 255])
    
#     # 获取图像尺寸
#     height, width = img_with_white_border.shape

#     # 绘制修剪线 (一条黑色线条，距离标记四周 trim_line_offset_px 像素)
#     # 上边线
#     cv2.line(img_with_white_border, 
#              (trim_line_offset_px, trim_line_offset_px), 
#              (width - trim_line_offset_px, trim_line_offset_px), 
#              (0, 0, 0), 2)

#     # 下边线
#     cv2.line(img_with_white_border, 
#              (trim_line_offset_px, height - trim_line_offset_px), 
#              (width - trim_line_offset_px, height - trim_line_offset_px), 
#              (0, 0, 0), 2)

#     # 左边线
#     cv2.line(img_with_white_border, 
#              (trim_line_offset_px, trim_line_offset_px), 
#              (trim_line_offset_px, height - trim_line_offset_px), 
#              (0, 0, 0), 2)

#     # 右边线
#     cv2.line(img_with_white_border, 
#              (width - trim_line_offset_px, trim_line_offset_px), 
#              (width - trim_line_offset_px, height - trim_line_offset_px), 
#              (0, 0, 0), 2)

#     # 保存带修剪线的标记图像
#     cv2.imwrite(f'aruco_marker_{marker_id}_with_trim_line.png', img_with_white_border)
#     print(f"已保存带修剪线的 ArUco 标记 ID {marker_id} 为 aruco_marker_{marker_id}_with_trim_line.png")

# # 生成 ID 为 0 的 ArUco
# generate_aruco_marker_with_trim_line(marker_id=1)

# import cv2
# import cv2.aruco as aruco

# def generate_aruco_marker_cm(marker_id, cm_size=19, dpi=300, dictionary=aruco.DICT_4X4_50):
#     # 轉換 cm → pixel
#     pixels_per_cm = dpi / 2.54
#     pixel_size = int(cm_size * pixels_per_cm)

#     # 取得 ArUco 字典
#     aruco_dict = aruco.getPredefinedDictionary(dictionary)

#     # 產生 ArUco 圖像
#     img = aruco.generateImageMarker(aruco_dict, marker_id, pixel_size)

#     # 儲存圖像（使用 dpi 附加資訊）
#     filename = f'aruco_marker_{marker_id}_{cm_size}cm_{dpi}dpi.png'
#     cv2.imwrite(filename, img)
#     print(f"✅ 已儲存 {cm_size}cm ArUco ID={marker_id} 圖像為 {filename}")

# # 使用範例：產生 ID=0 的 19 公分 ArUco
# generate_aruco_marker_cm(marker_id=1, cm_size=19)

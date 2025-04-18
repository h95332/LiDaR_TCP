import open3d as o3d
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import struct


def load_ply_with_pose(filepath):
    with open(filepath, 'rb') as f:
        header_lines = []
        while True:
            line = f.readline().decode('utf-8').strip()
            header_lines.append(line)
            if line == "end_header":
                break

        vertex_count = 0
        for line in header_lines:
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
                break

        poses = {
            'camera': [0.0] * 16,
            'lidar': [0.0] * 16,
            'world': [0.0] * 16,
        }

        for line in header_lines:
            if line.startswith("comment"):
                parts = line.split()
                if len(parts) == 3:
                    tag, value = parts[1], parts[2]
                    for key in poses.keys():
                        if tag.startswith(key + "_"):
                            index = int(tag.split("_")[1])
                            poses[key][index] = float(value)

        # 姿態轉為 4x4
        cam_pose = np.array(poses['camera'], dtype=np.float32).reshape((4, 4))
        lidar_pose = np.array(poses['lidar'], dtype=np.float32).reshape((4, 4))
        world_pose = np.array(poses['world'], dtype=np.float32).reshape((4, 4))

        # 點雲資料
        points = np.fromfile(f, dtype=np.float32, count=vertex_count * 3).reshape((-1, 3))
        return points, cam_pose, lidar_pose, world_pose


def create_axis(transform, size=0.2):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    frame.transform(transform)
    return frame


def open_ply_file():
    filepath = filedialog.askopenfilename(
        filetypes=[("PLY files", "*.ply")],
        title="選擇一個 PLY 檔案"
    )
    if not filepath:
        return

    if not os.path.exists(filepath):
        messagebox.showerror("錯誤", f"找不到檔案：{filepath}")
        return

    try:
        # 自訂載入：點雲與姿態
        with open(filepath, 'rb') as f:
            header_lines = []
            while True:
                line = f.readline().decode('utf-8').strip()
                header_lines.append(line)
                if line == "end_header":
                    break

            vertex_count = 0
            for line in header_lines:
                if line.startswith("element vertex"):
                    vertex_count = int(line.split()[-1])
                    break

            poses = {
                'camera': [0.0] * 16,
                'lidar': [0.0] * 16,
                'world': [0.0] * 16,
            }
            for line in header_lines:
                if line.startswith("comment"):
                    parts = line.split()
                    if len(parts) == 3:
                        tag, value = parts[1], parts[2]
                        for key in poses.keys():
                            if tag.startswith(key + "_"):
                                index = int(tag.split("_")[1])
                                poses[key][index] = float(value)

            cam_pose = np.array(poses['camera'], dtype=np.float32).reshape((4, 4))
            lidar_pose = np.array(poses['lidar'], dtype=np.float32).reshape((4, 4))
            world_pose = np.array(poses['world'], dtype=np.float32).reshape((4, 4))
            points = np.fromfile(f, dtype=np.float32, count=vertex_count * 3).reshape((-1, 3))

        if len(points) == 0:
            messagebox.showwarning("警告", "點雲是空的")
            return

        print("📌 Camera Pose:\n", cam_pose)
        print("📌 LiDAR Pose:\n", lidar_pose)
        print("📌 World Pose:\n", world_pose)

        # 點雲
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # ➕ 加入三組姿態軸（不改你原本 UI）
        cam_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5).transform(cam_pose)
        lidar_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5).transform(lidar_pose)
        world_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5).transform(world_pose)

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=os.path.basename(filepath), width=1280, height=720)
        vis.add_geometry(pcd)
        vis.add_geometry(cam_axis)
        vis.add_geometry(lidar_axis)
        vis.add_geometry(world_axis)

        opt = vis.get_render_option()
        opt.point_size = 1.0
        opt.background_color = np.asarray([0.05, 0.05, 0.05])
        opt.show_coordinate_frame = True

        vis.run()
        vis.destroy_window()

    except Exception as e:
        messagebox.showerror("讀取錯誤", f"無法讀取檔案：\n{str(e)}")


# GUI 主程式
if __name__ == '__main__':
    root = tk.Tk()
    root.title("PLY 點雲與姿態檢視器")
    root.geometry("300x130")
    root.resizable(False, False)

    label = tk.Label(root, text="請選擇包含姿態資訊的 .ply 檔案", font=("Helvetica", 11))
    label.pack(pady=15)

    btn = tk.Button(root, text="開啟 PLY 檔案", font=("Helvetica", 12), command=open_ply_file)
    btn.pack()

    root.mainloop()

import open3d as o3d
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import os

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
        pcd = o3d.io.read_point_cloud(filepath)
        if pcd.is_empty():
            messagebox.showwarning("警告", "點雲是空的")
            return

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=os.path.basename(filepath), width=1280, height=720)
        vis.add_geometry(pcd)

        opt = vis.get_render_option()
        opt.point_size = 1.0  # 讓點更粗
        opt.background_color = np.asarray([0.05, 0.05, 0.05])  # 深背景
        opt.show_coordinate_frame = True

        vis.run()
        vis.destroy_window()

    except Exception as e:
        messagebox.showerror("讀取錯誤", f"無法讀取檔案：\n{str(e)}")

# GUI 主程式
if __name__ == '__main__':
    root = tk.Tk()
    root.title("PLY 點雲檢視器")
    root.geometry("300x120")
    root.resizable(False, False)

    label = tk.Label(root, text="請選擇 .ply 檔案", font=("Helvetica", 12))
    label.pack(pady=15)

    btn = tk.Button(root, text="開啟 PLY", font=("Helvetica", 12), command=open_ply_file)
    btn.pack()

    root.mainloop()

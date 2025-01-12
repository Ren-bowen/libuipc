import igl
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

def render_obj(file_path, ax):
    """
    渲染一个 OBJ 文件
    :param file_path: OBJ 文件路径
    :param ax: Matplotlib 3D 轴对象
    """
    V, F = igl.read_triangle_mesh(file_path)
    mesh = Poly3DCollection(V[F], alpha=0.7, edgecolor='k')
    mesh.set_facecolor((0.6, 0.8, 1, 0.6))  # 设置颜色
    ax.add_collection3d(mesh)

    # 调整坐标轴范围
    scale = V.flatten()
    ax.auto_scale_xyz(scale, scale, scale)

# 设置文件夹路径，包含所有 OBJ 文件
obj_folder = "path/to/obj/files"

# 获取所有 OBJ 文件
obj_files = [f for f in os.listdir(obj_folder) if f.endswith('.obj')]

# 渲染每个文件
for i, obj_file in enumerate(obj_files):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    file_path = os.path.join(obj_folder, obj_file)
    print(f"Rendering {file_path} ({i + 1}/{len(obj_files)})")

    render_obj(file_path, ax)
    plt.title(f"Rendering: {obj_file}")
    plt.show()

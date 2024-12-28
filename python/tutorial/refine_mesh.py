import trimesh
import numpy as np
# 读取 OBJ 文件
mesh = trimesh.load_mesh('/root/libuipc/python/mesh/init_mesh_dress3.obj')

# 获取顶点和面信息
vertices = mesh.vertices  
faces = mesh.faces 

p = np.array([0.010794, 0.397191, -1.059256])

n = -1
for i in range(len(vertices)):
    if np.linalg.norm(vertices[i] - p) < 1e-4:  # 找到与p相等的顶点
        n = i
        vertices = np.delete(vertices, i, axis=0)  # 删除顶点
        break

print(n)
if (n > -1):
    faces = faces[~np.any(faces == n, axis=1)]  # 删除受影响的面
    faces[faces > n] -= 1  # 修正索引大于被删除的索引n

    new_mesh = trimesh.Trimesh(vertices, faces)
    new_mesh.export('/root/libuipc/python/mesh/init_mesh_dress3.obj')
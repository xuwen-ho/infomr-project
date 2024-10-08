import numpy as np
import open3d as o3d


def normalize_shape(mesh):
    #Position normalization
    # First get the barycenter position of the shape
    vertices = np.asarray(mesh.vertices)

    # p_updated = vertex - barycenter
    barycenter = np.mean(vertices, axis=0)
    vertices_centered = vertices - barycenter

    # Alignment normalization
    # Apply principal component analysis (PCA)

    #Compute new AABB after alignment normalization
    covariance_matrix = np.cov(vertices_centered.T)

    # Compute scaling factor
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # 使用特征向量对顶点进行对齐
    vertices_aligned = vertices_centered @ eigenvectors

    # Step 3: 尺度标准化 (将模型缩放至单位立方体)
    # 获取对齐后的模型的轴对齐边界框（AABB）
    min_bound = np.min(vertices_aligned, axis=0)
    max_bound = np.max(vertices_aligned, axis=0)
    aabb_size = max_bound - min_bound

    # 计算缩放因子，确保最大维度为 1
    scaling_factor = 1.0 / np.max(aabb_size)

    # 缩放顶点
    vertices_scaled = vertices_aligned * scaling_factor

    # 更新 mesh 的顶点
    mesh.vertices = o3d.utility.Vector3dVector(vertices_scaled)

    return mesh


# 读取一个 3D 网格文件 (OBJ/PLY/STL等格式)
mesh = o3d.io.read_triangle_mesh("path_to_your_mesh_file.obj")

# 进行标准化
normalized_mesh = normalize_shape(mesh)

# 保存标准化后的模型
o3d.io.write_triangle_mesh("normalized_mesh.obj", normalized_mesh)

# 可视化标准化后的模型
o3d.visualization.draw_geometries([normalized_mesh])

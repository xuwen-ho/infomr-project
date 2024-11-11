import open3d as o3d
import pymeshlab
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.spatial import distance
import os
import csv


# 使用 pymeshlab 修复网格的孔洞
def preprocess_mesh(mesh_file):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(mesh_file)

    # 检测并修复孔洞
    print(f"Checking and closing holes in {mesh_file}...")
    ms.meshing_close_holes()

    # 修复非流形边
    ms.meshing_repair_non_manifold_edges()

    # 保存修复后的网格
    fixed_mesh_file = f"{mesh_file.split('.')[0]}_fixed.obj"
    ms.save_current_mesh(fixed_mesh_file)

    return fixed_mesh_file

# 使用三角形面片计算封闭网格的体积
def calculate_volume(mesh):
    volume = 0.0
    vertex_matrix = mesh.vertex_matrix()
    face_matrix = mesh.face_matrix()

    for face in face_matrix:
        v0 = vertex_matrix[face[0]]
        v1 = vertex_matrix[face[1]]
        v2 = vertex_matrix[face[2]]

        # 计算三角形的体积贡献
        volume += np.dot(v0, np.cross(v1, v2))

    return abs(volume) / 6.0

# 使用 PyMeshLab 和 Open3D 计算全局特征
def compute_global_descriptors(mesh_file):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(mesh_file)

    # 获取表面积
    measures = ms.get_geometric_measures()
    area = measures.get('surface_area', None)
    if area is None:
        raise ValueError("Unable to find 'surface_area' in the geometric measures.")

    # 提取原始网格的顶点矩阵
    vertices = ms.current_mesh().vertex_matrix()

    # 计算所有顶点之间的最大距离（直径）
    max_dist = np.max(distance.cdist(vertices, vertices))

    # 计算有向包围盒 (OBB) 体积
    mesh_o3d = o3d.io.read_triangle_mesh(mesh_file)
    obb = mesh_o3d.get_oriented_bounding_box()
    obb_volume = obb.volume()

    # 使用自定义方法计算网格体积
    volume = calculate_volume(ms.current_mesh())

    # 计算凸包体积
    convex_hull, _ = mesh_o3d.compute_convex_hull()
    convex_hull_volume = convex_hull.get_volume()

    # 计算紧凑度
    compactness = (area ** 3) / (36 * np.pi * (volume ** 2))

    # 计算凸度：网格体积除以凸包体积
    convexity = volume / convex_hull_volume

    # 计算偏心率
    cov_matrix = np.cov(vertices.T)
    eigenvalues = np.linalg.eigvals(cov_matrix)
    eccentricity = np.max(eigenvalues) / np.min(eigenvalues)

    return {
        "Surface Area": area,
        "Compactness": compactness,
        "Rectangularity": volume / obb_volume,
        "Diameter": max_dist,
        "Convexity": convexity,
        "Eccentricity": eccentricity
    }

# 保存直方图并进行归一化
def normalize_histogram(hist):
    total = sum(hist)
    return [h / total for h in hist]

def save_histogram(data, title, filename):
    output_dir = os.path.dirname(filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure()
    # 直接使用已经归一化的 data
    plt.bar(range(len(data)), data, width=1.0, edgecolor="black")
    plt.title(title)
    plt.xlabel('Bins')
    plt.ylabel('Normalized Frequency')
    plt.savefig(filename)
    plt.close()


# 计算局部特征分布
def compute_shape_descriptors(mesh, filename_prefix, num_samples=160000, num_bins=100):
    vertices = np.asarray(mesh.vertices)
    centroid = np.mean(vertices, axis=0)

    # 生成随机样本，计算A3, D1, D2, D3, D4
    a3_angles, d1_distances, d2_distances, d3_areas, d4_volumes = [], [], [], [], []

    for _ in range(num_samples):
        idx = np.random.choice(len(vertices), 4, replace=False)
        v0, v1, v2, v3 = vertices[idx[0]], vertices[idx[1]], vertices[idx[2]], vertices[idx[3]]

        # A3: 角度
        a = np.linalg.norm(v1 - v0)
        b = np.linalg.norm(v2 - v0)
        c = np.linalg.norm(v2 - v1)
        # angle = np.arccos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))
        # a3_angles.append(angle)

        # Skip invalid triangles or ones that would cause numerical issues
        if a == 0 or b == 0:
            continue
        # Compute cosine using the law of cosines
        cos_angle = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)
        # Handle numerical stability
        if cos_angle > 1:
            cos_angle = 1
        elif cos_angle < -1:
            cos_angle = -1       
        angle = np.arccos(cos_angle)        
        # Only append valid angles
        if not np.isnan(angle):
            a3_angles.append(angle)

        # D1: 重心到顶点的距离
        d1_distances.append(np.linalg.norm(v0 - centroid))

        # D2: 两个随机顶点的距离
        d2_distances.append(np.linalg.norm(v0 - v1))

        # D3: 三角形面积
        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        d3_areas.append(np.sqrt(area))

        # D4: 四面体体积
        volume = np.abs(np.dot(np.cross(v1 - v0, v2 - v0), v3 - v0)) / 6.0
        d4_volumes.append(np.cbrt(volume))

    # 归一化局部特征分布
    a3_angles = normalize_histogram(np.histogram(a3_angles, bins=num_bins, density=False)[0].tolist())
    d1_distances = normalize_histogram(np.histogram(d1_distances, bins=num_bins, density=False)[0].tolist())
    d2_distances = normalize_histogram(np.histogram(d2_distances, bins=num_bins, density=False)[0].tolist())
    d3_areas = normalize_histogram(np.histogram(d3_areas, bins=num_bins, density=False)[0].tolist())
    d4_volumes = normalize_histogram(np.histogram(d4_volumes, bins=num_bins, density=False)[0].tolist())

    # # 保存直方图
    save_histogram(a3_angles, 'A3 Angle Distribution', f'{filename_prefix}_A3.png')
    save_histogram(d1_distances, 'D1 Distance Distribution', f'{filename_prefix}_D1.png')
    save_histogram(d2_distances, 'D2 Distance Distribution', f'{filename_prefix}_D2.png')
    save_histogram(d3_areas, 'D3 Triangle Area Distribution', f'{filename_prefix}_D3.png')
    save_histogram(d4_volumes, 'D4 Tetrahedron Volume Distribution', f'{filename_prefix}_D4.png')

    # 返回归一化后的数据字典
    return {
        'A3_hist': a3_angles,
        'D1_hist': d1_distances,
        'D2_hist': d2_distances,
        'D3_hist': d3_areas,
        'D4_hist': d4_volumes
    }


# 保存特征到 CSV 文件
def save_features_to_csv(features, csv_file="shape_features.csv"):
    fieldnames = ["Shape Name", "Class"] + list(features.keys())[2:]

    # 检查 CSV 文件是否存在
    file_exists = os.path.isfile(csv_file)

    # 使用写入模式更新 CSV 文件
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # 如果文件不存在，写入标题
        if not file_exists:
            writer.writeheader()

        # 写入模型的特征
        writer.writerow(features)

# 从 CSV 文件加载并打印特定形状的特征
def load_and_print_features(shape_name, csv_file="shape_features.csv"):
    with open(csv_file, mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row["Shape Name"] == shape_name:
                print(json.dumps(row, indent=4))
                return
        print(f"No features found for shape '{shape_name}'.")


# 处理单个3D模型
def process_single_model(mesh_file, output_prefix):
    try:
        print(f"Processing {mesh_file}...")
        # 使用 os.path.split 分割路径，获取文件名
        directory, filename = os.path.split(mesh_file)
        # 去掉文件的扩展名，获取模型名称
        shape_name = os.path.splitext(filename)[0]
        # 获取上级目录的名称作为类别
        shape_class = os.path.basename(directory)

        # 修复孔洞并确保网格一致
        # fixed_mesh_file = preprocess_mesh(mesh_file)

        # 加载修复后的网格
        mesh_o3d = o3d.io.read_triangle_mesh(mesh_file)
        mesh_o3d.compute_vertex_normals()

        # 计算全局特征
        global_descriptors = compute_global_descriptors(mesh_file)

        # 计算局部特征并保存直方图
        shape_descriptors = compute_shape_descriptors(mesh_o3d, output_prefix)

        features = {"Shape Name": shape_name, "Class": shape_class, **global_descriptors, **shape_descriptors}
        save_features_to_csv(features)
        print(f"Features saved for {shape_name}.")

        # 示例：加载并打印指定形状的特征
        load_and_print_features(shape_name)
    except Exception as e:
        print(f"Error processing {mesh_file}: {e}")

def process_dataset(input_folder, output_folder):
    for dirpath, _, filenames in os.walk(input_folder):
        for file in filenames:  # file should be like D0001.obj
            if file.endswith('.obj'):  # filter .obj
                input_path = os.path.join(input_folder, file)
                print(input_path)

                output_prefix = os.path.join(output_folder, file)
                print(output_prefix)
                try:
                    process_single_model(input_path, output_prefix)
                except Exception as e:
                    print(f"Error processing {file}: {e}")

# 示例：处理单个模型
def main():
    input_folder = 'Normalized/Drum'
    output_folder = 'output/Drum'
    mesh_file = "Normalized/AircraftBuoyant/m1337.obj"  # 替换为实际的3D模型路径
    output_prefix = "output/model"  # 输出文件前缀

    # process_dataset(input_folder, output_folder)
    # 创建一个球体
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    sphere.compute_vertex_normals()

    # 保存球体为 OBJ 文件
    o3d.io.write_triangle_mesh("Test/sphere.obj", sphere)

    # 创建一个立方体
    cube = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
    cube.compute_vertex_normals()

    # 保存立方体为 OBJ 文件
    o3d.io.write_triangle_mesh("Test/cube.obj", cube)

    # 创建一个圆柱体
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.5, height=2.0)
    cylinder.compute_vertex_normals()

    # 保存圆柱体为 OBJ 文件
    o3d.io.write_triangle_mesh("Test/cylinder.obj", cylinder)
    process_single_model("Test/sphere.obj", output_prefix)
    process_single_model("Test/cube.obj", output_prefix)
    process_single_model("Test/cylinder.obj", output_prefix)
if __name__ == "__main__":
    main()

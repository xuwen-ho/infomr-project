import re
import os
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import open3d as o3d
import pymeshlab as pml
import vedo
# Step 2: Preprocessing and cleaning
# Set up a class to store model info
class Model:
    def __init__(self, name, path, type, mesh, nVertices, nFaces, typeFaces, boungdingBox):
        self.name = name
        self.path = path
        self.type = type
        self.mesh = mesh
        self.nVertices = nVertices
        self.nFaces = nFaces
        self.typeFaces = typeFaces
        self.boundingBox = boungdingBox


# #Step 2.1: Analyzing a single shape
# # Output for each shape:
# #
# # the class of the shape
# # the number of faces and vertices of the shape
# # the type of faces (e.g. only triangles, only quads, mixes of triangles and quads)
# # the axis-aligned 3D bounding box of the shapes
#
# # file_path = "ShapeDatabase_INFOMR/AquaticAnimal/m54.obj" #copy the root path
# #
# # mesh = vedo.load(file_path)
# # vertices = mesh.vertices
# # num_vertices = vertices.shape[0]
# # num_faces = len(mesh.vertices)
# # print(num_vertices)
# # num_vertices_of_faces = mesh.count_vertices()
# # type_of_faces = np.unique(num_vertices_of_faces)
# # bounding_box = mesh.bounds()
#
# root = 'Normalized'
# #Step 2.2: Statistics over the whole database
# def generateCSV(dir):
#     models = [] #list to store models in the database
#     for dirpath, dirnames, filenames in os.walk(dir):
#         for file in filenames:  #file should be like D0001.obj
#             if file.endswith('.obj'): #filter .obj
#                 file_name = file[:-4] #Get the file name
#                 file_path = os.path.join(dirpath, file) # Get the file path
#                 folder_name = os.path.basename(dirpath) #Get the class of shapes
#                 mesh = vedo.load(file_path) #Load the model
#                 vertices = mesh.vertices # Get the
#                 num_vertices = vertices.shape[0]
#                 num_faces = len(mesh.cells)
#                 num_vertices_of_faces = mesh.count_vertices()
#                 type_of_faces = np.unique(num_vertices_of_faces)
#                 bounding_box = mesh.bounds()
#                 model = Model(file_name, file_path, folder_name, mesh, num_vertices, num_faces, type_of_faces, bounding_box)
#                 models.append(model)
#                 print(len(models))
#
#     # Save in a CSV file
#     with open(dir+'.csv', 'w', newline='') as file:
#         writer = csv.writer(file)
#         # Write the top line
#         writer.writerow(['File name', 'Class', 'Number of faces', 'Number of vertices','Type of faces', 'Bounding box'])
#         # Write the data on each line
#         for model in models:
#             print(model.name)
#             writer.writerow([model.name, model.type, model.nFaces, model.nVertices, model.typeFaces, model.boundingBox])
#
#     print(f"结果已保存到 {dir}.csv")

#Step 2.3: Resampling outliers

# print(pml.filter_list())

# # 定义一个辅助函数来打印网格状态
def print_mesh_status(step,mesh):
    num_vertices = mesh.current_mesh().vertex_number()
    num_faces = mesh.current_mesh().face_number()
    print(f'[{step}] 顶点数: {num_vertices}, 面数: {num_faces}')

# 设置日志文件
LOG_FILE = 'non_manifold_models.log'

def log_non_manifold_model(name, folder):
    with open(LOG_FILE, 'a') as log_file:
        log_file.write(f"{folder}/{name} is non-manifold after processing.\n")
#
# #Remove some outliers
# def process_outliers(name, folder, path):
#     ms = pml.MeshSet()
#     ms.load_new_mesh(path)
#     print_mesh_status('原始模型', ms)
#
#     try:
#         # 移除重复顶点
#         ms.apply_filter('meshing_remove_duplicate_vertices')
#         print_mesh_status('移除重复顶点', ms)
#
#         # 去除重复的面
#         ms.apply_filter('meshing_remove_duplicate_faces')
#         print_mesh_status('移除重复面', ms)
#
#         # 去除零面积面
#         ms.apply_filter('meshing_remove_null_faces')
#         print_mesh_status('移除空面', ms)
#
#         # 移除折叠面
#         ms.apply_filter('meshing_remove_folded_faces')
#         print_mesh_status('移除折叠面', ms)
#
#         # 修复非流形顶点
#         ms.apply_filter('meshing_repair_non_manifold_vertices')
#         print_mesh_status('修复非流形顶点', ms)
#
#         # 修复非流形边
#         ms.apply_filter('meshing_repair_non_manifold_edges')
#         print_mesh_status('修复非流形边', ms)
#
#         # 去除孤立的顶点
#         ms.apply_filter('meshing_remove_unreferenced_vertices')
#         print_mesh_status('移除未引用的顶点', ms)
#
#         # 填充孔洞
#         ms.apply_filter('meshing_close_holes', maxholesize=100)
#         print_mesh_status('填充孔洞', ms)
#
#         # 移除孤立片面
#         ms.apply_filter('meshing_remove_connected_component_by_face_number', mincomponentsize=10)
#         print_mesh_status('移除孤立片面', ms)
#
#         # 重新计算法线
#         ms.apply_filter('compute_normal_per_vertex')
#         print_mesh_status('计算顶点法线', ms)
#         ms.apply_filter('compute_normal_per_face')
#         print_mesh_status('计算面法线', ms)
#         # 确保目录存在
#         tmp_dir = f'tmp/{folder}'
#         if not os.path.exists(tmp_dir):
#             os.makedirs(tmp_dir)
#         tmp_path = f'tmp/{folder}/{name}.obj'
#         ms.save_current_mesh(tmp_path)
#
#         v_mesh = vedo.load(tmp_path)
#         v_mesh.non_manifold_faces(remove=True, tol='auto')
#         vedo.write(v_mesh, tmp_path)
#         ms.load_new_mesh(tmp_path)
#         if not v_mesh.is_manifold():
#             print(f"Warning: {name} is still not manifold after repair.")
#             with open("repair_log.txt", "a") as log_file:
#                 log_file.write(f"{name} in {folder} is still not manifold after repair.\n")
#         else:
#             ms.apply_filter('meshing_re_orient_faces_coherently')
#             print_mesh_status('重新调整面法线方向', ms)
#         # # 检查网格是否为流形
#         # num_non_manifold_edges = ms.apply_filter('compute_selection_by_non_manifold_edges_per_face')
#         # num_non_manifold_vertices = ms.apply_filter('compute_selection_by_non_manifold_per_vertex')
#         # if num_non_manifold_edges == 0 and num_non_manifold_vertices == 0:
#         #     # 如果没有非流形的顶点和边，进行法线一致性调整
#         #     ms.apply_filter('meshing_re_orient_faces_coherently')
#         #     print_mesh_status('重新调整面法线方向', ms)
#         # else:
#         #     print(f"Warning: {name} is not manifold; skipping re-orientation of face normals.")
#         #     log_non_manifold_model(name, folder)
#
#     except Exception as e:
#         print(f"Error processing filters: {e}")
#
#     # 确保目录存在
#     output_dir = f'Outliers_Removed/{folder}'
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#
#     # 保存处理后的网格
#     ms.save_current_mesh(f'{output_dir}/{name}.obj')
#
#     # 输出保存后的网格文件路径
#     print(f'Repaired mesh saved as {output_dir}/{name}.obj')
#
#     # 最终统计信息
#     print_mesh_status('最终处理后的网格', ms)

# #Remove some outliers
def process_outliers0(name, folder, path):
    ms = pml.MeshSet()
    ms.load_new_mesh(path)
    print_mesh_status('原始模型', ms)

    try:
        # 移除重复顶点
        ms.apply_filter('meshing_remove_duplicate_vertices')
        print_mesh_status('移除重复顶点', ms)

        # 去除重复的面
        ms.apply_filter('meshing_remove_duplicate_faces')
        print_mesh_status('移除重复面', ms)

        # 去除零面积面
        ms.apply_filter('meshing_remove_null_faces')
        print_mesh_status('移除空面', ms)

        # 移除折叠面
        ms.apply_filter('meshing_remove_folded_faces')
        print_mesh_status('移除折叠面', ms)

        # 修复非流形顶点
        ms.apply_filter('meshing_repair_non_manifold_vertices')
        print_mesh_status('修复非流形顶点', ms)

        # 修复非流形边
        ms.apply_filter('meshing_repair_non_manifold_edges')
        print_mesh_status('修复非流形边', ms)

        # 确保目录存在
        tmp_dir = f'tmp/{folder}'
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        tmp_path = f'tmp/{folder}/{name}.obj'
        ms.save_current_mesh(tmp_path)

        v_mesh = vedo.load(tmp_path)
        v_mesh.non_manifold_faces(remove=True, tol='auto')
        vedo.write(v_mesh, tmp_path)
        ms.load_new_mesh(tmp_path)
        # 去除孤立的顶点
        ms.apply_filter('meshing_remove_unreferenced_vertices')
        print_mesh_status('移除未引用的顶点', ms)

        ms.apply_filter('meshing_merge_close_vertices')
        print_mesh_status('合并相近的顶点', ms)


        # 重新计算法线
        ms.apply_filter('compute_normal_per_vertex')
        print_mesh_status('计算顶点法线', ms)
        ms.apply_filter('compute_normal_per_face')
        print_mesh_status('计算面法线', ms)

        ms.apply_filter('meshing_re_orient_faces_by_geometry')
        print_mesh_status('根据几何形状重新调整面部法线', ms)
        # 确保目录存在
        tmp_dir = f'tmp/{folder}'
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        tmp_path = f'tmp/{folder}/{name}.obj'
        ms.save_current_mesh(tmp_path)

        v_mesh = vedo.load(tmp_path)
        v_mesh.non_manifold_faces(remove=True, tol='auto')
        vedo.write(v_mesh, tmp_path)
        ms.load_new_mesh(tmp_path)
        if not v_mesh.is_manifold():
            print(f"Warning: {name} is still not manifold after repair.")
            with open("repair_log.txt", "a") as log_file:
                log_file.write(f"{name} in {folder} is still not maM>?nifold after repair.\n")
        else:
            ms.apply_filter('meshing_re_orient_faces_coherently')
            print_mesh_status('重新调整面法线方向', ms)
            # ms.apply_filter('meshing_invert_face_orientation')
            # print_mesh_status('反转面方向', ms)

    except Exception as e:
        print(f"Error processing filters: {e}")

    # 确保目录存在
    output_dir = f'Outliers_Removed/{folder}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存处理后的网格
    ms.save_current_mesh(f'{output_dir}/{name}.obj')

    # 输出保存后的网格文件路径
    print(f'Repaired mesh saved as {output_dir}/{name}.obj')

    # 最终统计信息
    print_mesh_status('最终处理后的网格', ms)

root = 'ShapeDatabase_INFOMR'
for dirpath, dirnames, filenames in os.walk(root):
    for file in filenames:  #file should be like D0001.obj
        if file.endswith('.obj'): #filter .obj
            file_name = file[:-4] #Get the file name
            file_path = os.path.join(dirpath, file) # Get the file path
            folder_name = os.path.basename(dirpath) #Get the class of shapes
            process_outliers0(file_name, folder_name, file_path)

print('Done')

# Resample
# 参数
# input_folder = "Outliers_Removed0"  # 包含所有模型的文件夹路径（低多边形或高多边形模型）
# output_folder = "Resampled0"  # 输出结果保存路径
# target_vertex_count = 5000  # 目标顶点数量
# max_iterations = 20  # 最大迭代次数，避免无限迭代
# tolerance = 500  # 容忍范围，接近目标顶点数时停止迭代
# # 设置日志文件
# LOG_FILE = 'Resampling.log'
# # 确保输出文件夹存在
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)
#
# # 批量处理函数，用于简化或细化网格到目标顶点数
# def process_mesh_to_target_vertices():
#     for dirpath, dirnames, filenames in os.walk(input_folder):
#         for file in filenames:  #file should be like D0001.obj
#             if file.endswith('.obj'): #filter .obj
#                 folder_name = os.path.basename(dirpath) #Get the class of shapes
#                 input_path = os.path.join(input_folder, folder_name, file)
#                 output_path = os.path.join(output_folder, folder_name, file)
#                 tmp_path = os.path.join('tmp', folder_name, file)
#
#                 # 确保目录存在
#                 tmp_dir = os.path.join('tmp', folder_name)
#                 if not os.path.exists(tmp_dir):
#                     os.makedirs(tmp_dir)
#
#                 # 确保目录存在
#                 output_dir = os.path.join(output_folder, folder_name)
#                 if not os.path.exists(output_dir):
#                     os.makedirs(output_dir)
#
#                 try:
#                     # 创建 MeshSet 对象并加载网格
#                     ms = pml.MeshSet()
#                     # Initialize Open3D and Vedo versions of the mesh
#                     mesh_o3d = o3d.io.read_triangle_mesh(input_path)
#                     original_vertex_count = len(mesh_o3d.vertices)
#                     print(f"Original vertices: {original_vertex_count}")
#
#                     if abs(len(mesh_o3d.vertices) - target_vertex_count) > tolerance:
#                         # Use PyMeshLab to redistribute vertices evenly
#                         # ms.load_new_mesh(input_path)  # Load updated mesh in PyMeshLab
#                         # ms.apply_filter('meshing_isotropic_explicit_remeshing', adaptive= True)
#                         # ms.save_current_mesh(tmp_path)  # Save redistributed mesh
#                         # mesh_o3d = o3d.io.read_triangle_mesh(tmp_path)
#                         # current_vertex_count = len(mesh_o3d.vertices)
#                         # print(f"Remeshed vertices: {current_vertex_count}")
#                         # Loop to achieve target vertex count
#                         iterations = 0
#                         while abs(len(mesh_o3d.vertices) - target_vertex_count) > tolerance:
#                             current_vertex_count = len(mesh_o3d.vertices)
#                             if len(mesh_o3d.vertices) < target_vertex_count:
#                                 # Upsampling with Open3D
#                                 print(f"Upsampling {file} to increase vertices...")
#                                 mesh_o3d = mesh_o3d.subdivide_midpoint(number_of_iterations=1)
#                                 o3d.io.write_triangle_mesh(tmp_path, mesh_o3d)
#                             else:
#                                 # Downsampling with Vedo
#                                 print(f"Downsampling {file} to reduce vertices...")
#                                 v_mesh = vedo.load(tmp_path)
#                                 v_mesh = v_mesh.decimate_pro(n=target_vertex_count)
#                                 vedo.write(v_mesh, tmp_path)
#
#                             # Reload in Open3D to continue processing
#                             mesh_o3d = o3d.io.read_triangle_mesh(tmp_path)
#                             iterations += 1
#                             print(f"Iteration {iterations}: Vertex count after redistribution is {len(mesh_o3d.vertices)}")
#
#                             if len(mesh_o3d.vertices) == current_vertex_count:  # Prevent infinite loops
#                                 print(f"Resampling no longer better for {file}.")
#                                 break
#
#                             if  iterations == max_iterations:
#                                 print(f"Iterations meet maximum for {file}.")
#                                 with open(LOG_FILE, 'a') as log_file:
#                                     log_file.write(f"Iterations meet maximum for {file} with {len(mesh_o3d.vertices)} vertices in {folder_name}: {e}\n")
#                                 break
#
#                     # Save the final processed mesh_o3d
#                     o3d.io.write_triangle_mesh(output_path, mesh_o3d)
#                     print(f"Final mesh saved as {output_path} (final vertices: {len(mesh_o3d.vertices)})")
#
#                 except Exception as e:
#                     print(f"Error processing {file}: {e}")
#                     with open("processing_errors_log.txt", "a") as log_file:
#                         log_file.write(f"Error processing {file} in {folder_name}: {e}\n")
#
#
# # 批量处理所有模型，将顶点数调整到目标数量（5000左右）
# process_mesh_to_target_vertices()
#
# print("Batch processing finished.")


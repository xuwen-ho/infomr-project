import vedo
import re
import os
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import open3d as o3d
import pymeshlab as pml

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
# #Step 2.2: Statistics over the whole database
root = 'Normalized'
models = [] #list to store models in the database
for dirpath, dirnames, filenames in os.walk(root):
    for file in filenames:  #file should be like D0001.obj
        if file.endswith('.obj'): #filter .obj
            file_name = file[:-4] #Get the file name
            file_path = os.path.join(dirpath, file) # Get the file path
            folder_name = os.path.basename(dirpath) #Get the class of shapes
            mesh = vedo.load(file_path) #Load the model
            vertices = mesh.vertices # Get the
            num_vertices = vertices.shape[0]
            num_faces = len(mesh.cells)
            num_vertices_of_faces = mesh.count_vertices()
            type_of_faces = np.unique(num_vertices_of_faces)
            bounding_box = mesh.bounds()
            model = Model(file_name, file_path, folder_name, mesh, num_vertices, num_faces, type_of_faces, bounding_box)
            models.append(model)
            print(len(models))

# Save in a CSV file
with open('Normalized.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    # Write the top line
    writer.writerow(['File name', 'Class', 'Number of faces', 'Number of vertices','Type of faces', 'Bounding box'])
    # Write the data on each line
    for model in models:
        print(model.name)
        writer.writerow([model.name, model.type, model.nFaces, model.nVertices, model.typeFaces, model.boundingBox])

print(f"结果已保存到 {'Normalized.csv'}")

#Step 2.3: Resampling outliers
# 创建 MeshSet 对象并加载网格

# print(pml.filter_list())
# 定义一个辅助函数来打印网格状态
# def print_mesh_status(step,mesh):
#     num_vertices = mesh.current_mesh().vertex_number()
#     num_faces = mesh.current_mesh().face_number()
#     print(f'[{step}] 顶点数: {num_vertices}, 面数: {num_faces}')
#
# #Remove some outliers
# def process_outliers(name, folder, path):
#     ms = pml.MeshSet()
#     ms.load_new_mesh(path)
#     print_mesh_status('原始模型',ms)
#
#     # 移除重复顶点
#     ms.apply_filter('meshing_remove_duplicate_vertices')
#     print_mesh_status('移除重复顶点',ms)
#
#     # 去除重复的面
#     ms.apply_filter('meshing_remove_duplicate_faces')
#     print_mesh_status('移除重复面',ms)
#
#     # 去除零面积面
#     ms.apply_filter('meshing_remove_null_faces')
#     print_mesh_status('移除空面',ms)
#
#     # 移除折叠面
#     ms.apply_filter('meshing_remove_folded_faces')
#     print_mesh_status('移除折叠面',ms)
#
#     # 修复非流形顶点
#     ms.apply_filter('meshing_repair_non_manifold_vertices')
#     print_mesh_status('修复非流形顶点' ,ms)
#
#     # 修复非流形边
#     ms.apply_filter('meshing_repair_non_manifold_edges')
#     print_mesh_status('修复非流形边' ,ms)
#
#     # 去除孤立的顶点
#     ms.apply_filter('meshing_remove_unreferenced_vertices')
#     print_mesh_status('移除未引用的顶点',ms)
#
#     # 填充孔洞
#     # ms.apply_filter('meshing_close_holes', maxholesize=100)
#     # print_mesh_status('填充孔洞')
#
#     # # 移除孤立片面
#     # ms.apply_filter('meshing_remove_connected_component_by_face_number', mincomponentsize=10)
#     # print_mesh_status('移除孤立片面',ms)
#
#     # 重新计算法线
#     ms.apply_filter('compute_normal_per_vertex')
#     print_mesh_status('计算顶点法线',ms)
#     ms.apply_filter('compute_normal_per_face')
#     print_mesh_status('计算面法线',ms)
#
#     #检查并翻转面的法线一致性
#     ms.apply_filter('meshing_re_orient_faces_coherently')
#     print_mesh_status('重新调整面法线方向',ms)
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
#     print(f'Repaired mesh saved as Outliers_Removed/{folder}/{name}.obj')
#
# root = 'ShapeDatabase_INFOMR'
# for dirpath, dirnames, filenames in os.walk(root):
#     for file in filenames:  #file should be like D0001.obj
#         if file.endswith('.obj'): #filter .obj
#             file_name = file[:-4] #Get the file name
#             file_path = os.path.join(dirpath, file) # Get the file path
#             folder_name = os.path.basename(dirpath) #Get the class of shapes
#             process_outliers(file_name, folder_name, file_path)
#
# print('Done')

#Resample
# # 参数
# input_folder = "Outliers_Removed"  # 包含所有模型的文件夹路径（低多边形或高多边形模型）
# output_folder = "Resampled"  # 输出结果保存路径
# target_vertex_count = 5000  # 目标顶点数量
# max_iterations = 10  # 最大迭代次数，避免无限迭代
# tolerance = 200  # 容忍范围，接近目标顶点数时停止迭代
#
# # 确保输出文件夹存在
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)
#
#
# # 批量处理函数，用于简化或细化网格到目标顶点数
# def process_mesh_to_target_vertices():
#     for dirpath, dirnames, filenames in os.walk(input_folder):
#         for file in filenames:  #file should be like D0001.obj
#             if file.endswith('.obj'): #filter .obj
#                 folder_name = os.path.basename(dirpath) #Get the class of shapes
#                 input_path = os.path.join(input_folder, folder_name, file)
#                 output_path = os.path.join(output_folder, folder_name, file)
#
#                 # 确保目录存在
#                 output_dir = os.path.join(output_folder, folder_name)
#                 if not os.path.exists(output_dir):
#                     os.makedirs(output_dir)
#
#                 try:
#                     # 创建 MeshSet 对象并加载网格
#                     ms = pml.MeshSet()
#                     ms.load_new_mesh(input_path)
#
#                     # 获取当前顶点数
#                     original_vertex_count = ms.current_mesh().vertex_number()
#                     print(f"Processing {file} (original vertices: {original_vertex_count})...")
#
#                     # 动态调整顶点数以接近目标顶点数
#                     if abs(original_vertex_count - target_vertex_count) <= tolerance:
#                         # 顶点数已接近目标，直接保存
#                         print(f"{file} already has vertex count close to target. Saving without changes.")
#                         ms.save_current_mesh(output_path)
#
#                     else:
#                         # 根据顶点数进行简化或上采样
#                         iterations = 0
#                         while abs(
#                                 ms.current_mesh().vertex_number() - target_vertex_count) > tolerance and iterations < max_iterations:
#                             current_vertex_count = ms.current_mesh().vertex_number()
#
#                             if current_vertex_count > target_vertex_count:
#                                 print(f"Simplifying {file} to reduce vertices...")
#                                 ms.meshing_decimation_quadric_edge_collapse(targetvertnum=target_vertex_count)
#                             else:
#                                 print(f"Upsampling {file} to increase vertices...")
#                                 ms.meshing_surface_subdivision_loop(iterations=1)
#
#                             iterations += 1
#                             new_vertex_count = ms.current_mesh().vertex_number()
#                             print(f"Iteration {iterations}: New vertex count is {new_vertex_count}")
#
#                         # 保存处理后的网格
#                         ms.save_current_mesh(output_path)
#                         print(f"Final mesh saved as {output_path} (final vertices: {new_vertex_count})")
#
#                 except Exception as e:
#                         print(f"Error processing {file}: {e}")
#
#
# # 批量处理所有模型，将顶点数调整到目标数量（5000左右）
# process_mesh_to_target_vertices()
#
# print("Batch processing finished.")


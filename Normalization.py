import numpy as np
import open3d as o3d
import copy
import os

def normalize_shape(mesh):
    # Step 2.5: Position normalization
    # First get the barycenter position of the shape
    vertices = np.asarray(mesh.vertices)

    # p_updated = vertex - barycenter
    barycenter = np.mean(vertices, axis=0)
    vertices_centered = vertices - barycenter

    # Step 3.1: Alignment normalization using PCA
    # Compute a covariance matrix of the vertex positions
    covariance_matrix = np.cov(vertices_centered.T)

    # Compute Eigenvectors and Eigenvalues of the covariance matrix & sort Eigenvectors by Eigenvalue
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Align the vertices using the eigenvectors
    vertices_aligned = vertices_centered @ eigenvectors

    # Step 3.1: Flipping test
    ## Calculate the sign of the mass distribution along each axis (X, Y, Z)
    signs = np.sign(np.sum(vertices_aligned, axis=0))  # 计算 X, Y, Z 轴上质量分布
    for i in range(3):  # Iterate over each axis (X, Y, Z)
        if signs[i] < 0:
            vertices_aligned[:, i] *= -1  # Flip the axis if needed

    # Step 2.5: Scale normalization
    # Calculate the axis-aligned bounding box (AABB) of the aligned shape
    min_bound = np.min(vertices_aligned, axis=0)
    max_bound = np.max(vertices_aligned, axis=0)
    aabb_size = max_bound - min_bound

    # Compute the scaling factor to ensure the largest dimension is 1
    scaling_factor = 1.0 / np.max(aabb_size)

    # Scale the vertices to fit within a unit cube
    vertices_scaled = vertices_aligned * scaling_factor

    # Update the mesh with the scaled vertices
    mesh.vertices = o3d.utility.Vector3dVector(vertices_scaled)

    return mesh


input_folder = "Resampled"
output_folder = "Normalized"

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Batch process mesh
for dirpath, dirnames, filenames in os.walk(input_folder):
    for file in filenames:  #file should be like D0001.obj
        if file.endswith('.obj'): #filter .obj
            folder_name = os.path.basename(dirpath) #Get the class of shapes
            input_path = os.path.join(input_folder, folder_name, file)
            output_path = os.path.join(output_folder, folder_name, file)

            # 确保目录存在
            output_dir = os.path.join(output_folder, folder_name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            try:
                # 读取一个 3D 网格文件 (OBJ/PLY/STL等格式)
                mesh = o3d.io.read_triangle_mesh(input_path)

                # # Create a copy of the original mesh for visualization purposes
                # original_mesh = copy.deepcopy(mesh)

                # Perform the normalization
                normalized_mesh = normalize_shape(mesh)

                # Save the normalized mesh
                o3d.io.write_triangle_mesh(output_path, normalized_mesh)

                # # Move the original mesh slightly to the left for side-by-side comparison
                # original_mesh.translate([-2, 0, 0])

                # # Move the normalized mesh slightly to the right for side-by-side comparison
                # normalized_mesh.translate([2, 0, 0])
                #
                # # Visualize the original and normalized meshes together
                # o3d.visualization.draw_geometries([original_mesh, normalized_mesh],
                #                                   window_name="Original vs Normalized Mesh",
                #                                   width=800, height=600)

                print(f"Processed and saved: {output_path}")

            except Exception as e:
                print(f"Error processing {file}: {e}")





import vedo
import re
import os
import numpy as np
import csv

#Step 2: Preprocessing and cleaning
#Set up a class to store model info
class Model:
    def __init__(self, name, path, type, mesh, nVertices, nFaces, typeFaces, boungdingBox):
        self.name = name
        self.path = path
        self.type = type
        self.mesh = mesh
        self.nVertices = nVertices
        self.nFaces = nVertices
        self.typeFaces = typeFaces
        self.boundingBox = boungdingBox


#Step 2.1: Analyzing a single shape
# Output for each shape:
#
# the class of the shape
# the number of faces and vertices of the shape
# the type of faces (e.g. only triangles, only quads, mixes of triangles and quads)
# the axis-aligned 3D bounding box of the shapes

# file_path = "ShapeDatabase_INFOMR/AquaticAnimal/m54.obj" #copy the root path
#
# mesh = vedo.load(file_path)
# vertices = mesh.vertices
# num_vertices = vertices.shape[0]
# num_faces = len(mesh.vertices)
# print(num_vertices)
# num_vertices_of_faces = mesh.count_vertices()
# type_of_faces = np.unique(num_vertices_of_faces)
# bounding_box = mesh.bounds()

#Step 2.2: Statistics over the whole database
root = 'ShapeDatabase_INFOMR'
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
with open('ShapeDatabase_INFOMR.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    # 写入 CSV 文件的表头
    writer.writerow(['File name', 'Class', 'Number of faces', 'Number of vertices','Type of faces', 'Bounding box'])
    # 写入每一行数据
    for model in models:
        print(model.name)
        writer.writerow([model.name, model.type, model.nFaces, model.nVertices, model.typeFaces, model.boundingBox])

print(f"结果已保存到 {'ShapeDatabase_INFOMR.csv'}")

#Step 2.3: Resampling outliers

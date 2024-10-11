import os
from vedo import *

"""
Step 2.4 Normalisation
Step process: 
1) Translate barycenter to origin
2) Compute principle axes and align with coordinate frame
3) Flip based on moment test
4) Scale to unit volume
"""

def process_obj_files(directory, normalisationStepList):
    # Ensure the directory exists
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return

    # Iterate through all .obj files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".obj"):
            filepath = os.path.join(directory, filename)
            
            # Load the mesh
            mesh = load(filepath)
            
            # Find the barycenter
            barycenter = mesh.center_of_mass()
            print(f"Barycenter of {filename}: {barycenter}")
            
            # Shift the center to the origin
            mesh.shift(-barycenter)

            ellipsoid = pca_ellipsoid(mesh)
            print("axis 1 size:", ellipsoid.va)
            print("axis 2 size:", ellipsoid.vb)
            print("axis 3 size:", ellipsoid.vc)
            print("axis 1 direction:", ellipsoid.axis1)
            print("axis 2 direction:", ellipsoid.axis2)
            print("axis 3 direction:", ellipsoid.axis3)

            axisSizeList = np.array([ellipsoid.va, ellipsoid.vb, ellipsoid.vc])
            axisList = np.array([ellipsoid.axis1, ellipsoid.axis2, ellipsoid.axis3])

            sizeIndex = axisSizeList.argsort()[::-1]
            axisSizeList = axisSizeList[sizeIndex]
            axisList = axisList[sizeIndex]

            print(axisSizeList, axisList)
            # Save the modified mesh
            # output_filepath = os.path.join(directory, f"centered_{filename}")
            # write(mesh, output_filepath)

            angle_x = np.degrees(np.arccos(np.dot(ellipsoid.axis1, [1,0,0])))
            angle_y = np.degrees(np.arccos(np.dot(ellipsoid.axis2, [0,1,0])))
            angle_z = np.degrees(np.arccos(np.dot(ellipsoid.axis3, [0,0,1])))
            print(f"  X-axis: {angle_x:.2f}°")
            print(f"  Y-axis: {angle_y:.2f}°")
            print(f"  Z-axis: {angle_z:.2f}°")

            
            
            # print(f"Processed and saved: {output_filepath}")

            xAxisArrow = Arrow(ellipsoid.center, ellipsoid.center + ellipsoid.axis1)
            yAxisArrow = Arrow(ellipsoid.center, ellipsoid.center + ellipsoid.axis2, c="g4")
            zAxisArrow = Arrow(ellipsoid.center, ellipsoid.center + ellipsoid.axis3, c="b4")
            triad = Assembly(xAxisArrow, yAxisArrow, zAxisArrow)

            # mesh.reorient(1, vector(ellipsoid.axis2[0],ellipsoid.axis2[1],ellipsoid.axis2[2]), xyplane=True)
            # mesh.reorient(initaxis=1, newaxis=ellipsoid.axis1)

            # 2.4.4: Scale it to within unit volume
            bounds = mesh.bounds()
            dimensions = [
                abs(bounds[1] - bounds[0]),  # X-axis
                abs(bounds[3] - bounds[2]),  # Y-axis
                abs(bounds[5] - bounds[4])   # Z-axis
            ]
            longest_length = max(dimensions)
            mesh.scale(1/longest_length)

            # 2.4.4 Test: Ensure within unit volume
            print(mesh.bounds())

            show(
                mesh, triad, axes=2,).close()

# Specify the directory containing the .obj files
directory = "Resampled/car"



# Process the files
process_obj_files(directory, [])


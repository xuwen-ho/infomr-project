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

            
            
            # print(f"Processed and saved: {output_filepath}")

            a1 = Arrow(ellipsoid.center, ellipsoid.center + ellipsoid.axis1)
            a2 = Arrow(ellipsoid.center, ellipsoid.center + ellipsoid.axis2, c="b4")
            a3 = Arrow(ellipsoid.center, ellipsoid.center + ellipsoid.axis3, c="g4")
            triad = Assembly(a1, a2, a3) 

            # mesh.reorient(1, vector(ellipsoid.axis2[0],ellipsoid.axis2[1],ellipsoid.axis2[2]), xyplane=True)
            mesh.reorient(initaxis=1, newaxis=ellipsoid.axis1)

            show(
                mesh, triad, axes=2,).close()

# Specify the directory containing the .obj files
directory = "Resampled/car"



# Process the files
process_obj_files(directory, [])


import os
from vedo import Mesh, load, write

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
            
            # Save the modified mesh
            output_filepath = os.path.join(directory, f"centered_{filename}")
            write(mesh, output_filepath)
            
            print(f"Processed and saved: {output_filepath}")

# Specify the directory containing the .obj files
directory = "Resampled/Guitar"



# Process the files
process_obj_files(directory)


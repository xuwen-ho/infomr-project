import os
import numpy as np
from vedo import Plotter, load, Text2D, Button
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from scipy.spatial import distance
import math
import open3d as o3d
import pymeshlab
from scipy.stats import wasserstein_distance
import ast  # Add this import for safely evaluating string representations of lists

# Import functions from feature_extraction.py
from feature_extraction import (
    compute_global_descriptors,
    compute_shape_descriptors,
    load_standardization_stats,
    preprocess_mesh,
    standardize_features
)
import pandas as pd

class ShapeComparisonGUI:
    def __init__(self):
        print("Initializing ShapeComparisonGUI...")  # Debug print
        # self.database_path = os.path.join(os.getcwd(), "C:\Users\User\Downloads\Normalized")
        self.database_path = r"C:\Users\User\Downloads\Normalized"
        print(f"Looking for database in: {self.database_path}")  # Debug print
        self.shapes_dict = {}  # Dictionary to store loaded shapes
        self.current_shape = None
        self.selected_shape_path = None
        self.K = 5

        self.features_db = None
        self.load_precomputed_features()
        
        # Initialize the plotter with interactive flag
        self.plt = Plotter(
            N=6,  # Two viewports
            axes=2,
            interactive=True,
            size=(1200, 800),  # Larger window size
            bg='black'
        )
        
        # Add buttons with correct function binding
        select_button = self.plt.add_button(
            fnc=self.show_shape_selector,  # Debug print
            pos=(0.2, 0.05),
            states=['Select Shape'],
            c=['w'],
            bc=['dg'],  # dark green
            size=12,
        )

        search_button = self.plt.add_button(
            fnc=self.search_similar_shapes,
            pos=[0.8, 0.05],
            states=['Search Similar'],
            c=['w'],
            bc=['db'],
            size=12,
        )
        
        status = Text2D(
            "Select a shape to begin",
            pos='top-middle',
            s=0.8,
            c='white',
        )
        
        # Store UI elements as instance variables
        self.select_btn = select_button
        self.search_btn = search_button
        self.status_text = status
        
        # Add UI elements to the plotter
        # self.plt.add([select_button, compare_button, status])
        self.plt.add([status])
        print("Adding buttons to plotter...")  # Debug print
        
        # Load shapes from database
        self.load_database()

    def load_database(self):
        """Load all shapes from the database directory"""
        print("Loading database...")  # Debug print
        try:
            for folder in os.listdir(self.database_path):
                folder_path = os.path.join(self.database_path, folder)
                if os.path.isdir(folder_path):
                    for file in os.listdir(folder_path):
                        if file.endswith(('.obj', '.ply', '.stl')):  # Add more extensions if needed
                            file_path = os.path.join(folder_path, file)
                            # Store the full path and category (folder name)
                            self.shapes_dict[file_path] = {
                                'category': folder,
                                'name': file
                            }
            
            print(f"Loaded {len(self.shapes_dict)} shapes")  # Debug print
            self.status_text.text(f"Loaded {len(self.shapes_dict)} shapes in database")
        except Exception as e:
            print(f"Error loading database: {str(e)}")  # Debug print
            self.status_text.text(f"Error loading database: {str(e)}")

    def show_shape_selector(self, obj, ename):
        """Create a Tkinter window to select shapes from the database"""
        print("Opening shape selector window...")  # Debug print
        
        # Create a new Tkinter window
        root = tk.Tk()
        select_window = tk.Toplevel(root)
        select_window.title("Select Shape")
        select_window.geometry("400x450")

        # Create a treeview to display shapes organized by category
        tree = ttk.Treeview(select_window)
        tree.pack(fill='both', expand=True, padx=5, pady=5)

        # Create scrollbar
        scrollbar = ttk.Scrollbar(select_window, orient="vertical", command=tree.yview)
        scrollbar.pack(side='right', fill='y')

        tree.configure(yscrollcommand=scrollbar.set)

        # Configure treeview columns
        tree["columns"] = ("name")
        tree.column("#0", width=120, minwidth=120)
        tree.column("name", width=280, minwidth=280)

        # Configure column headings
        tree.heading("#0", text="Category")
        tree.heading("name", text="Shape Name")

        # Organize shapes by category
        categories = {}
        for path, info in self.shapes_dict.items():
            category = info['category']
            if category not in categories:
                categories[category] = []
            categories[category].append((path, info['name']))

        # Populate the treeview
        for category, shapes in categories.items():
            category_id = tree.insert("", "end", text=category)
            for path, name in shapes:
                tree.insert(category_id, "end", values=(name,), tags=(path,))

        # Bind selection event
        def on_select(event):
            print("Shape selected in tree view")  # Debug print
            selected_items = tree.selection()
            if selected_items:
                item = selected_items[0]
                if tree.item(item)['values']:  # Check if it's a shape and not a category
                    file_path = tree.item(item)['tags'][0]
                    self.load_selected_shape(file_path)
                    select_window.destroy()
                    root.destroy()

        tree.bind('<<TreeviewSelect>>', on_select)

        # Add a close button
        close_btn = ttk.Button(
            select_window,
            text="Close",
            command=lambda: [select_window.destroy(), root.destroy()]
        )
        close_btn.pack(pady=5)

        # Start the Tkinter main loop
        root.withdraw()  # Hide the root window
        root.mainloop()

    def compute_feature_distances(self, features1, features2):
        """
        Compute distances between different feature types separately
        Returns distances for global features and each histogram separately
        """
        # Define indices for different feature types based on the feature vector structure
        GLOBAL_FEATURES_END = 6  # First 6 are global features
        HIST_SIZE = 100  # Size of each histogram
        
        # Extract global features
        global_feat1 = features1[:GLOBAL_FEATURES_END]
        global_feat2 = features2[:GLOBAL_FEATURES_END]
        
        # Initialize distances dictionary
        distances = {
            'global': self.compute_euclidean_distance(global_feat1, global_feat2),
            'A3': 0.0,
            'D1': 0.0,
            'D2': 0.0,
            'D3': 0.0,
            'D4': 0.0
        }
        
        # Compute histogram distances
        start_idx = GLOBAL_FEATURES_END
        for hist_name in ['A3', 'D1', 'D2', 'D3', 'D4']:
            end_idx = start_idx + HIST_SIZE
            hist1 = features1[start_idx:end_idx]
            hist2 = features2[start_idx:end_idx]
            distances[hist_name] = self.compute_emd_distance(hist1, hist2)
            start_idx = end_idx
            
        return distances

    def standardize_distances(self, all_pairwise_distances):
        """
        Standardize distances for each feature type based on the distribution
        of distances in the database
        """
        standardized = {}
        
        # For each feature type
        for feat_type in all_pairwise_distances[0].keys():
            # Get all distances for this feature type
            distances = [d[feat_type] for d in all_pairwise_distances]
            
            # Compute mean and standard deviation
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            
            if std_dist == 0:
                print(f"Warning: Zero standard deviation for {feat_type} distances")
                std_dist = 1.0
                
            # Store standardization parameters
            standardized[feat_type] = {
                'mean': mean_dist,
                'std': std_dist
            }
            
        return standardized

    def compute_weighted_distance(self, features1, features2, standardization_params):
        """
        Compute weighted distance between two feature vectors
        """
        # Compute distances for each feature type
        distances = self.compute_feature_distances(features1, features2)
        
        # Standardize each distance
        weighted_distances = {}
        for feat_type, dist in distances.items():
            mean = standardization_params[feat_type]['mean']
            std = standardization_params[feat_type]['std']
            weighted_distances[feat_type] = (dist - mean) / std
        
        # Combine standardized distances
        # You can adjust these weights based on the importance of each feature type
        weights = {
            'global': 0.3,  # Global features
            'A3': 0.14,     # Angle histogram
            'D1': 0.14,     # Distance to center histogram
            'D2': 0.14,     # Distance between vertices histogram
            'D3': 0.14,     # Triangle area histogram
            'D4': 0.14      # Tetrahedron volume histogram
        }
        
        total_distance = sum(
            weights[feat_type] * weighted_distances[feat_type]
            for feat_type in weighted_distances
        )
        
        return total_distance

    # Add method to load pre-computed features
    def load_precomputed_features(self):
        """Load pre-computed features from CSV file"""
        try:
            self.features_db = pd.read_csv("shape_features_1.csv")
            # Convert string representations of histograms back to numpy arrays
            hist_columns = ['A3_hist', 'D1_hist', 'D2_hist', 'D3_hist', 'D4_hist']
            for col in hist_columns:
                self.features_db[col] = self.features_db[col].apply(
                    lambda x: np.array(ast.literal_eval(x))
                )
            print(f"Loaded {len(self.features_db)} pre-computed feature vectors")
        except Exception as e:
            print(f"Error loading pre-computed features: {str(e)}")
            messagebox.showerror("Error", f"Failed to load feature database: {str(e)}")

    # Update search method to use pre-computed features
    def search_similar_shapes(self, obj, ename):
        """Search using pre-computed features from CSV"""
        if not self.current_shape:
            print("No shape selected")
            self.status_text.text("Please select a shape first")
            messagebox.showwarning("Warning", "Please select a shape first")
            return
        
        try:
            # Extract features from query shape only
            query_features = self.extract_features(self.current_shape)
            
            # Convert query features to same format as database
            global_features = query_features[:6]
            hist_features = query_features[6:]
            
            # Split histograms
            hist_size = 100
            query_hists = {
                'A3_hist': hist_features[:hist_size],
                'D1_hist': hist_features[hist_size:2*hist_size],
                'D2_hist': hist_features[2*hist_size:3*hist_size],
                'D3_hist': hist_features[3*hist_size:4*hist_size],
                'D4_hist': hist_features[4*hist_size:5*hist_size]
            }
            
            # Calculate distances to all shapes in database
            distances = []
            for idx, row in self.features_db.iterrows():
                if os.path.basename(self.selected_shape_path) == row['Shape Name']:
                    continue
                    
                # Calculate weighted distance
                global_dist = self.compute_euclidean_distance(
                    global_features,
                    [row['Surface Area'], row['Compactness'], row['Rectangularity'],
                     row['Diameter'], row['Convexity'], row['Eccentricity']]
                )
                # print('calculating emd distance')
                # global_dist = self.compute_emd_distance(
                #     global_features,
                #     [row['Surface Area'], row['Compactness'], row['Rectangularity'],
                #      row['Diameter'], row['Convexity'], row['Eccentricity']]
                # )
                
                # Calculate histogram distances
                hist_distances = {}
                for hist_name in ['A3_hist', 'D1_hist', 'D2_hist', 'D3_hist', 'D4_hist']:
                    hist_distances[hist_name] = self.compute_emd_distance(
                        query_hists[hist_name],
                        row[hist_name]
                    )
                
                # Apply weights (same as before)
                weights = {
                    'global': 0.14,
                    'A3': 0.14,
                    'D1': 0.14,
                    'D2': 0.14,
                    'D3': 0.14,
                    'D4': 0.14
                }
                
                total_distance = (
                    weights['global'] * global_dist +
                    weights['A3'] * hist_distances['A3_hist'] +
                    weights['D1'] * hist_distances['D1_hist'] +
                    weights['D2'] * hist_distances['D2_hist'] +
                    weights['D3'] * hist_distances['D3_hist'] +
                    weights['D4'] * hist_distances['D4_hist']
                )
                
                # Find the corresponding file path
                shape_path = os.path.join(
                    self.database_path,
                    row['Class'],
                    row['Shape Name']
                )
                distances.append((total_distance, shape_path))
            
            # Sort and display results
            distances.sort(key=lambda x: x[0])

            # Clear all viewports except the first one (query shape)
            for i in range(1, 6):  # Clear viewports 1-
                print("clear previous")
                self.plt.clear(at=i)
                # self.plt.remove(at=i)  # This ensures complete clearing of previous results
            
            # Display K most similar shapes
            for i in range(min(self.K, len(distances))):
                dist, path = distances[i]
                print("Loading best matching shapes:", path)
                shape = load(path + ".obj")

                shape.flat().lighting('plastic')
                
                # Clear and update viewport
                # self.plt.clear(at=i+1)
                self.plt.show(shape, at=i+1, interactive=False)
                
                similarity = 100 * np.exp(-max(0, dist))
                
                self.plt.add(Text2D(
                    f"Similarity: {similarity:.1f}%\n{os.path.basename(path)}", 
                    pos='top-left', 
                    s=0.8, 
                    c='w', 
                    bg='black'
                ), at=i+1)

            self.plt.show(self.current_shape, at=0, interactive=False)
            
            self.status_text.text(f"Found {self.K} most similar shapes")
            print("Shape search completed")
            
        except Exception as e:
            print(f"Error during shape search: {str(e)}")
            self.status_text.text(f"Error during search: {str(e)}")
            messagebox.showerror("Error", f"Search failed: {str(e)}")

    def load_selected_shape(self, file_path):
        """Load and display the selected shape"""

        # file_path = self.database_path
        print(f"Loading shape: {file_path}")  # Debug print
        try:
            # Load and display the selected shape
            self.selected_shape_path = file_path
            self.current_shape = load(file_path)
            
            self.current_shape.flat().lighting('plastic')

            # Clear and update the first viewport
            self.plt.clear(at=0)
            # self.plt.pop(at=0)    # Remove actors from the scene
            # self.plt.renderer(at=0).RemoveAllViewProps()  # Force remove all VTK props
            self.plt.show(self.current_shape, at=0, interactive=False)
            
            # self.plt.add(Text2D(
            #         f"Selected: {os.path.basename(file_path)}", 
            #         pos='top-left', 
            #         s=0.8, 
            #         c='w', 
            #         bg='black'
            #     ), at=0)
            self.status_text.text(f"Selected: {os.path.basename(file_path)}")
            print("Shape loaded successfully")  # Debug print
        except Exception as e:
            print(f"Error loading shape: {str(e)}")  # Debug print
            self.status_text.text(f"Error loading shape: {str(e)}")
            messagebox.showerror("Error", f"Failed to load shape: {str(e)}")

    def extract_features(self, mesh):
        """Extract geometric features from a mesh"""
        try:
            # Save the mesh temporarily to use with feature extraction functions
            temp_file = "temp_mesh.obj"
            mesh.write(temp_file)
            
            # Preprocess the mesh
            fixed_mesh_file = preprocess_mesh(temp_file)
            
            # Load with Open3D for shape descriptors
            mesh_o3d = o3d.io.read_triangle_mesh(fixed_mesh_file)
            mesh_o3d.compute_vertex_normals()
            
            # Get global descriptors
            global_features = compute_global_descriptors(fixed_mesh_file)
            stats_file = "standardization_stats_1.json"  
            means, stds = load_standardization_stats(stats_file)
            print(means, stds)
            global_descriptors = standardize_features(global_features, means, stds)
            print("standardized global descriptors", global_descriptors)

            # Get shape descriptors (local features)
            shape_features = compute_shape_descriptors(mesh_o3d, "temp_output")
            
            # Combine all features into a single vector
            feature_vector = np.array([
                global_descriptors['Surface Area'],
                global_descriptors['Compactness'],
                global_descriptors['Rectangularity'],
                global_descriptors['Diameter'],
                global_descriptors['Convexity'],
                global_descriptors['Eccentricity']
            ])
            print('combined features successfully')
            
            # Append histograms from shape descriptors
            for hist in shape_features.values():
                feature_vector = np.concatenate([feature_vector, hist])
            
            # Clean up temporary files
            os.remove(temp_file)
            if os.path.exists(fixed_mesh_file):
                os.remove(fixed_mesh_file)
                
            return feature_vector
            
        except Exception as e:
            print(f"Error in feature extraction: {str(e)}")
            raise

    # def normalize_features(self, features_list):
    #     """Normalize features using min-max normalization"""
    #     features_array = np.array(features_list)
    #     min_vals = np.min(features_array, axis=0)
    #     max_vals = np.max(features_array, axis=0)
        
    #     # Avoid division by zero
    #     range_vals = max_vals - min_vals
    #     range_vals[range_vals == 0] = 1
        
    #     normalized_features = (features_array - min_vals) / range_vals
    #     return normalized_features

    def compute_euclidean_distance(self, features1, features2):
        """Compute Euclidean distance between two feature vectors"""
        return distance.euclidean(features1, features2)

    def compute_manhattan_distance(self, features1, features2):
        """Compute Manhattan (L1) distance between two feature vectors"""
        return distance.cityblock(features1, features2)

    def compute_emd_distance(self, features1, features2):
        """
        Compute Earth Mover's Distance (Wasserstein distance) between feature vectors
        Note: This is particularly useful for comparing histograms in the feature vectors
        """
        # Ensure the features are properly normalized for EMD
        f1_norm = features1 / np.sum(features1)
        f2_norm = features2 / np.sum(features2)
        return wasserstein_distance(f1_norm, f2_norm)

    def run(self):
      """Start the GUI"""
      print("Starting GUI...")
      self.plt.show(interactive=True)

if __name__ == "__main__":
    print("Creating application instance...")  # Debug print
    app = ShapeComparisonGUI()
    print("Running application...")  # Debug print
    app.run()
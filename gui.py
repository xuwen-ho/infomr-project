import os
import numpy as np
from vedo import Plotter, load, Text2D, Button
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from scipy.spatial import distance
import math

class ShapeComparisonGUI:
    def __init__(self):
        print("Initializing ShapeComparisonGUI...")  # Debug print
        self.database_path = os.path.join(os.getcwd(), "Normalized")
        print(f"Looking for database in: {self.database_path}")  # Debug print
        self.shapes_dict = {}  # Dictionary to store loaded shapes
        self.current_shape = None
        self.selected_shape_path = None
        self.K = 5
        
        # Initialize the plotter with interactive flag
        self.plt = Plotter(
            N=6,  # Two viewports
            axes=1,
            interactive=True,
            size=(1200, 800),  # Larger window size
            bg='black'
        )
        
        # Add buttons with correct function binding
        select_button = self.plt.add_button(
            fnc=self.show_shape_selector,  # Debug print
            pos=(0.1, 0.05),
            states=['Select Shape'],
            c=['w'],
            bc=['dg'],  # dark green
            size=12,
        )

        search_button = self.plt.add_button(
            fnc=self.search_similar_shapes,
            pos=[0.9, 0.05],
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

    def search_similar_shapes(self, obj, ename):
        """Search for similar shapes based on geometric features"""
        if not self.current_shape:
            print("No shape selected")
            self.status_text.text("Please select a shape first")
            messagebox.showwarning("Warning", "Please select a shape first")
            return
        
        try:
            # Extract features from query shape
            query_features = self.extract_features(self.current_shape)
            
            # Compute features for all shapes in database and store distances
            distances = []
            all_features = []
            
            # First pass: extract all features
            for path in self.shapes_dict.keys():
                # Skip if it's the same file as the query
                if path == self.selected_shape_path:
                    print(f"Skipping query shape: {path}")
                    continue

                shape = load(path)
                features = self.extract_features(shape)
                all_features.append(features) 
            
            # Normalize all features (including query)
            normalized_features = self.normalize_features([query_features] + all_features)
            normalized_query = normalized_features[0]
            normalized_database = normalized_features[1:]
            
            # Compute distances
            for i, features in enumerate(normalized_database):
                dist = self.compute_euclidean_distance(normalized_query, features)
                distances.append((dist, list(self.shapes_dict.keys())[i]))
            
            # Sort by distance
            distances.sort(key=lambda x: x[0])
            
            # Clear all viewports except the first one (query shape)
            for i in range(1, 6):  # Clear viewports 1-5
                self.plt.clear(at=i)
            
            # Display K most similar shapes
            for i in range(min(self.K, len(distances))):
                dist, path = distances[i]
                shape = load(path)
                shape.flat().lighting('plastic')  # set light effect
                # Update viewport (add 1 to skip the query shape viewport)
                self.plt.show(shape, at=i+1, interactive=False)
                
                # Add distance information
                self.plt.add(Text2D(
                    f"Similarity: {(1 - dist)*100:.1f}%\n{os.path.basename(path)}", 
                    pos='top-left', 
                    s=0.8, 
                    c='w', 
                    bg='black'
                ), at=i+1)
            
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
            
            # Clear and update the first viewport
            self.plt.clear(at=0)
            self.current_shape.flat().lighting('plastic')  # set light effect
            self.plt.show(self.current_shape, at=0, interactive=False)
            
            self.status_text.text(f"Selected: {os.path.basename(file_path)}")
            print("Shape loaded successfully")  # Debug print
        except Exception as e:
            print(f"Error loading shape: {str(e)}")  # Debug print
            self.status_text.text(f"Error loading shape: {str(e)}")
            messagebox.showerror("Error", f"Failed to load shape: {str(e)}")

    # REPLACE BELOW PLACEHOLDERS
    def extract_features(self, mesh):
        """Extract geometric features from a mesh"""
        # Basic geometric features
        features = {}
        
        # Surface area
        features['surface_area'] = mesh.area()
        
        # Volume
        features['volume'] = mesh.volume()
        
        # Bounding box features
        bounds = mesh.bounds()
        features['bbox_length'] = bounds[1] - bounds[0]  # x dimension
        features['bbox_width'] = bounds[3] - bounds[2]   # y dimension
        features['bbox_height'] = bounds[5] - bounds[4]  # z dimension
        
        # Compactness (surface area³ / (36π * volume²))
        if features['volume'] > 0:
            features['compactness'] = (features['surface_area']**3) / (36 * math.pi * features['volume']**2)
        else:
            features['compactness'] = 0
        
        # Number of vertices and faces
        features['vertex_count'] = mesh.npoints
        features['face_count'] = mesh.ncells
        
        # Convert to numpy array in consistent order
        feature_vector = np.array([
            features['surface_area'],
            features['volume'],
            features['bbox_length'],
            features['bbox_width'],
            features['bbox_height'],
            features['compactness'],
            features['vertex_count'],
            features['face_count']
        ])
        
        return feature_vector

    def normalize_features(self, features_list):
        """Normalize features using min-max normalization"""
        features_array = np.array(features_list)
        min_vals = np.min(features_array, axis=0)
        max_vals = np.max(features_array, axis=0)
        
        # Avoid division by zero
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1
        
        normalized_features = (features_array - min_vals) / range_vals
        return normalized_features

    def compute_euclidean_distance(self, features1, features2):
        """Compute Euclidean distance between two feature vectors"""
        return distance.euclidean(features1, features2)
    # REPLACE UP TO HERE

    def run(self):
      """Start the GUI"""
      print("Starting GUI...")
      self.plt.show(interactive=True)

if __name__ == "__main__":
    print("Creating application instance...")  # Debug print
    app = ShapeComparisonGUI()
    print("Running application...")  # Debug print
    app.run()
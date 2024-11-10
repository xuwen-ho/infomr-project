import os
import numpy as np
from vedo import Plotter, load, Text2D, Button
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from scipy.spatial import distance
import math
from sklearn.neighbors import NearestNeighbors  # For ANN
from sklearn.manifold import TSNE  # For t-SNE
import matplotlib.pyplot as plt
import pandas as pd

class ShapeComparisonGUI:
    def __init__(self):
        print("Initializing ShapeComparisonGUI...")  # Debug print
        self.database_path = os.path.join(os.getcwd(), "Normalized")
        print(f"Looking for database in: {self.database_path}")  # Debug print
        self.shapes_dict = {}  # Dictionary to store loaded shapes
        self.current_shape = None
        self.selected_shape_path = None
        self.K = 5

         # Initialize feature storage
        self.feature_vectors = []
        self.shape_paths = []
        self.ann_model = None
        self.tsne_coords = None
        
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

        tsne_button = self.plt.add_button(
            fnc=self.show_tsne_plot,
            pos=(0.5, 0.05),
            states=['Show t-SNE'],
            c=['w'],
            bc=['dr'],  # dark red
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
        self.tsne_btn = tsne_button

        
        # Add UI elements to the plotter
        # self.plt.add([select_button, compare_button, status])
        self.plt.add([status])
        print("Adding buttons to plotter...")  # Debug print
        
        # Load shapes from database
        self.load_database()
        self.prepare_features()

    def prepare_features(self):
        """Extract features from all shapes and prepare ANN and t-SNE"""
        print("Preparing feature database...")
        
        # Extract features for all shapes
        for path in self.shapes_dict.keys():
            shape = load(path)
            features = self.extract_features(shape)
            self.feature_vectors.append(features)
            self.shape_paths.append(path)
        
        # Convert to numpy array
        self.feature_vectors = np.array(self.feature_vectors)
        
        # Normalize features
        self.feature_vectors = self.normalize_features(self.feature_vectors)
        
        # Initialize ANN model
        self.ann_model = NearestNeighbors(n_neighbors=self.K, algorithm='ball_tree')
        self.ann_model.fit(self.feature_vectors)
        
        # Compute t-SNE
        print("Computing t-SNE...")
        self.tsne_coords = TSNE(n_components=2, random_state=42).fit_transform(self.feature_vectors)
        
        print("Feature preparation completed")

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

    # def search_similar_shapes(self, obj, ename):
    #     """Search for similar shapes using ANN"""
    #     if not self.current_shape:
    #         print("No shape selected")
    #         self.status_text.text("Please select a shape first")
    #         messagebox.showwarning("Warning", "Please select a shape first")
    #         return
        
    #     try:
    #         # Extract and normalize query features
    #         query_features = self.extract_features(self.current_shape)
    #         query_features = self.normalize_features(np.array([query_features]))[0]
            
    #         # Find K nearest neighbors
    #         distances, indices = self.ann_model.kneighbors([query_features])
            
    #         # Clear viewports except the first one
    #         for i in range(1, 6):
    #             self.plt.clear(at=i)
            
    #         # Create results table
    #         results_data = []
            
    #         # Display K most similar shapes
    #         for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    #             if i >= self.K:
    #                 break
                    
    #             path = self.shape_paths[idx]
    #             shape = load(path)
    #             shape.flat().lighting('plastic')
                
    #             # Update viewport
    #             self.plt.show(shape, at=i+1, interactive=False)
                
    #             # Add similarity information
    #             similarity = (1 - dist) * 100
    #             self.plt.add(Text2D(
    #                 f"Similarity: {similarity:.1f}%\n{os.path.basename(path)}", 
    #                 pos='top-left', 
    #                 s=0.8, 
    #                 c='w', 
    #                 bg='black'
    #             ), at=i+1)
                
    #             # Add to results table
    #             results_data.append({
    #                 'Shape': os.path.basename(path),
    #                 'Category': self.shapes_dict[path]['category'],
    #                 'Similarity': f"{similarity:.1f}%"
    #             })
            
    #         # Display results table
    #         self.show_results_table(results_data)
            
    #         self.status_text.text(f"Found {self.K} most similar shapes")
    #         print("Shape search completed")
            
    #     except Exception as e:
    #         print(f"Error during shape search: {str(e)}")
    #         self.status_text.text(f"Error during search: {str(e)}")
    #         messagebox.showerror("Error", f"Search failed: {str(e)}")

    def show_results_table(self, results_data):
        """Display results in a table format"""
        table_window = tk.Toplevel()
        table_window.title("Search Results")
        table_window.geometry("600x300")
        
        # Create treeview
        tree = ttk.Treeview(table_window, columns=('Shape', 'Category', 'Similarity'), show='headings')
        tree.heading('Shape', text='Shape Name')
        tree.heading('Category', text='Category')
        tree.heading('Similarity', text='Similarity')
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(table_window, orient="vertical", command=tree.yview)
        scrollbar.pack(side='right', fill='y')
        tree.configure(yscrollcommand=scrollbar.set)
        
        # Add data to table
        for item in results_data:
            tree.insert('', 'end', values=(item['Shape'], item['Category'], item['Similarity']))
        
        tree.pack(fill='both', expand=True, padx=5, pady=5)

    def show_tsne_plot(self, obj, ename):
        """Display t-SNE visualization for all classes"""
        print("Displaying t-SNE plot...")
        
        # Create a figure with a specific size to accommodate the plot and legend
        fig = plt.figure(figsize=(16, 10))
        
        # Create a subplot that leaves room for the legend
        ax = plt.subplot(111)
        
        # Get categories and their counts
        category_counts = {}
        for info in self.shapes_dict.values():
            category = info['category']
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Get all categories sorted by count
        all_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        category_names = [cat for cat, _ in all_categories]
        
        # Generate distinct colors using a combination of colormaps and custom colors
        base_colors = []
        # Add colors from tab20
        base_colors.extend(plt.cm.tab20(np.linspace(0, 1, 20)))
        # Add colors from tab20b
        base_colors.extend(plt.cm.tab20b(np.linspace(0, 1, 20)))
        # Add colors from Set3
        base_colors.extend(plt.cm.Set3(np.linspace(0, 1, 12)))
        # Add colors from Pastel1
        base_colors.extend(plt.cm.Pastel1(np.linspace(0, 1, 9)))
        # Add colors from Paired
        base_colors.extend(plt.cm.Paired(np.linspace(0, 1, 12)))
        
        # Convert to list for easier manipulation
        base_colors = list(base_colors)
        
        # If we need more colors, generate them randomly but ensuring they're distinct
        while len(base_colors) < len(category_names):
            new_color = np.random.rand(3)
            # Ensure good contrast and saturation
            if min(new_color) > 0.2 and max(new_color) < 0.8:
                base_colors.append(np.append(new_color, 1.0))  # Add alpha channel
        
        # Shuffle colors for better contrast between adjacent categories
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(base_colors)
        
        # Create color dictionary
        category_colors = dict(zip(category_names, base_colors[:len(category_names)]))
        
        # Plot points
        for i, path in enumerate(self.shape_paths):
            category = self.shapes_dict[path]['category']
            color = category_colors[category]
            ax.scatter(self.tsne_coords[i, 0], self.tsne_coords[i, 1], 
                      c=[color], alpha=0.6, picker=True)
        
        # Highlight current shape if one is selected
        if self.selected_shape_path:
            idx = self.shape_paths.index(self.selected_shape_path)
            plt.scatter(self.tsne_coords[idx, 0], self.tsne_coords[idx, 1],
                       c='red', s=100, marker='*', label='Selected Shape')
        
        # Create legend elements
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=category_colors[cat], 
                                    label=f'{cat}', 
                                    markersize=8)
                         for cat, count in all_categories]
        
        # Calculate the optimal number of columns based on the number of categories
        ncols = min(5, max(3, len(category_names) // 10))
        
        # Create the legend with multiple columns
        legend = ax.legend(handles=legend_elements,
                         title="Shape Categories",
                         loc='center left',
                         bbox_to_anchor=(1, 0.5),
                         borderaxespad=0.5,
                         fontsize=8,
                         ncol=ncols,
                         columnspacing=1.0)
        
        # Set title and labels
        plt.title('t-SNE Visualization of Shape Features', pad=20)
        plt.xlabel('t-SNE dimension 1')
        plt.ylabel('t-SNE dimension 2')
        
        # Adjust layout to prevent label cutoff and accommodate legend
        plt.tight_layout()
        
        # Adjust the subplot to make room for the legend
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        
        # Enable point selection
        def on_pick(event):
            ind = event.ind[0]
            path = self.shape_paths[ind]
            shape_info = self.shapes_dict[path]
            plt.title(f'Selected: {os.path.basename(path)}\nCategory: {shape_info["category"]}')
            plt.draw()
        
        plt.gcf().canvas.mpl_connect('pick_event', on_pick)
        plt.show(block=True)
        
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
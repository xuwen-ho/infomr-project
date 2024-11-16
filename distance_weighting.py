import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist
import json

def parse_histogram_string(hist_str):
    """
    Parse a histogram string into a numpy array.
    
    Args:
        hist_str (str): String representation of histogram array "[0.123,0.456,...]"
        
    Returns:
        numpy.ndarray: Array of histogram values
    """
    # Remove brackets and split by comma
    try:
        # Remove brackets and split
        values = hist_str.strip('[]').split(',')
        # Convert to float array
        return np.array([float(x) for x in values])
    except:
        # Return zeros if there's any problem parsing
        return np.zeros(100)

def compute_feature_weights(csv_path):
    """
    Compute weights for all features in the shape dataset.
    
    Args:
        csv_path (str): Path to the CSV file containing shape features
        
    Returns:
        dict: Dictionary containing weights for all features
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Initialize weights dictionary
    weights = {}
    
    # 1. Process single-value features
    single_features = ['Surface Area', 'Compactness', 'Rectangularity', 
                      'Diameter', 'Convexity', 'Eccentricity']
    
    for feature in single_features:
        if feature in df.columns:
            # Compute standard deviation and weight
            std_dev = df[feature].std()
            weights[feature] = 1.0 / std_dev if std_dev > 0 else 1.0
    
    # 2. Process histogram features
    histogram_features = ['A3_hist', 'D1_hist', 'D2_hist', 'D3_hist', 'D4_hist']
    
    for hist_type in histogram_features:
        if hist_type in df.columns:
            # Convert string representations to arrays
            hist_arrays = np.array([parse_histogram_string(hist_str) 
                                  for hist_str in df[hist_type]])
            
            # Compute pairwise distances between all histograms
            distances = pdist(hist_arrays, metric='euclidean')
            
            # Compute standard deviation of distances
            std_dev = np.std(distances)
            weights[hist_type] = 1.0 / std_dev if std_dev > 0 else 1.0
    
    return weights

def save_weights(weights, output_path='feature_weights.json'):
    """
    Save computed weights to a JSON file
    """
    with open(output_path, 'w') as f:
        # Convert any numpy types to native Python types for JSON serialization
        weights_json = {k: float(v) for k, v in weights.items()}
        json.dump(weights_json, f, indent=4)

def print_weights_analysis(weights):
    """
    Print analysis of the computed weights
    """
    print("\nFeature Weights Analysis:")
    print("-" * 50)
    
    # Separate weights by type
    single_features = ['Surface Area', 'Compactness', 'Rectangularity', 
                      'Diameter', 'Convexity', 'Eccentricity']
    histogram_features = ['A3_hist', 'D1_hist', 'D2_hist', 'D3_hist', 'D4_hist']
    
    print("\nSingle-value Features:")
    for feature in single_features:
        if feature in weights:
            print(f"{feature:15s}: {weights[feature]:.6f}")
    
    print("\nHistogram Features:")
    for feature in histogram_features:
        if feature in weights:
            print(f"{feature:15s}: {weights[feature]:.6f}")
    
    # Print some statistics
    all_weights = list(weights.values())
    print("\nWeight Statistics:")
    print(f"Mean weight:   {np.mean(all_weights):.6f}")
    print(f"Median weight: {np.median(all_weights):.6f}")
    print(f"Min weight:    {np.min(all_weights):.6f}")
    print(f"Max weight:    {np.max(all_weights):.6f}")

def test_histogram_parsing():
    """
    Test function to verify histogram parsing
    """
    print("\nTesting histogram parsing:")
    test_str = "[0.0005454,0.45454,0.0]"  # truncated for brevity
    parsed = parse_histogram_string(test_str)
    print(f"Sample parsed histogram (first 3 values): {parsed[:3]}")
    print(f"Histogram length: {len(parsed)}")

def main():
    """
    Main function to compute and save weights
    """
    try:
        # Run a quick test of histogram parsing
        test_histogram_parsing()
        
        # Compute weights
        print("\nComputing weights...")
        weights = compute_feature_weights('shape_features_1.csv')
        
        # Save weights to JSON
        save_weights(weights)
        
        # Print analysis
        print_weights_analysis(weights)
        
        print("\nWeights have been successfully computed and saved to 'feature_weights.json'")
        
    except FileNotFoundError:
        print("Error: Could not find 'shape_features_1.csv'")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise  # This will show the full error trace

if __name__ == "__main__":
    main()
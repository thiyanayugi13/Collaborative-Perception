#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from pathlib import Path
import argparse
from sklearn.cluster import DBSCAN
import gc  # Garbage collection

class MemoryEfficientCleaner:
    """A memory-efficient version of the radar point cloud cleaner"""
    
    def __init__(self, input_path, output_path=None):
        self.input_path = input_path
        self.output_path = output_path
        
        # Default parameters - less aggressive
        self.params = {
            'boundary': {
                'x_min': -9.0,
                'x_max': 10.5,
                'y_min': -4.35,
                'y_max': 5.7,
                'buffer': 0.0    # Extended boundary
            },
            'height_threshold': {
                'min_height': -1.5,  # More permissive
                'max_height': 4.0    # More permissive
            },
            'statistical_outlier': {
                'k_neighbors': 20,    # More neighbors
                'std_ratio': 3.0      # Higher threshold
            },
            'voxel_grid': {
                'voxel_size': 0.05    # Smaller voxels
            }
        }
    
    def process(self):
        """Process a single file with memory-efficient operations"""
        print(f"Processing file: {self.input_path}")
        
        # Determine output path
        if not self.output_path:
            path_obj = Path(self.input_path)
            self.output_path = str(path_obj.parent / f"cleaned_{path_obj.name}")
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        # Create visualization path
        vis_path = os.path.join(os.path.dirname(self.output_path), 
                              f"cleaning_comparison_{os.path.basename(self.output_path)}.png")
        
        # Process data - load in chunks
        chunk_size = 10000  # Smaller chunks to reduce memory usage
        chunks = []
        
        # Count total rows first
        total_rows = pd.read_csv(self.input_path, nrows=2).shape[0]
        print(f"Estimated total rows: {total_rows}")
        
        # Process in chunks
        for chunk_idx, chunk in enumerate(pd.read_csv(self.input_path, chunksize=chunk_size)):
            print(f"Processing chunk {chunk_idx+1}...")
            
            # Apply simple filters that don't require full dataset
            chunk = self.filter_chunk(chunk)
            
            # Store processed chunk
            chunks.append(chunk)
            
            # Force garbage collection
            gc.collect()
        
        # Combine chunks
        if chunks:
            result = pd.concat(chunks, ignore_index=True)
            print(f"Combined {len(chunks)} chunks, resulting in {len(result)} points")
            
            # Generate simple visualization
            self.generate_visualization(result, vis_path)
            
            # Save result
            result.to_csv(self.output_path, index=False)
            print(f"Saved cleaned data to {self.output_path}")
            
            return True
        else:
            print("No data processed")
            return False
    
    def filter_chunk(self, chunk):
        """Apply memory-efficient filters to a chunk of data"""
        # Process each robot separately
        for robot_id in ['robot_1', 'robot_2']:
            # Column names
            x_col = f'{robot_id}_global_x_radar'
            y_col = f'{robot_id}_global_y_radar'
            z_col = f'{robot_id}_global_z_radar'
            
            # Skip if columns don't exist
            if not all(col in chunk.columns for col in [x_col, y_col, z_col]):
                continue
            
            # Create valid mask
            valid_mask = ~chunk[[x_col, y_col, z_col]].isna().any(axis=1)
            if not valid_mask.any():
                continue
            
            # 1. Apply boundary filter
            x_min = self.params['boundary']['x_min']
            x_max = self.params['boundary']['x_max']
            y_min = self.params['boundary']['y_min']
            y_max = self.params['boundary']['y_max']
            
            boundary_mask = (
                (chunk[x_col] >= x_min) & (chunk[x_col] <= x_max) &
                (chunk[y_col] >= y_min) & (chunk[y_col] <= y_max)
            ) | ~valid_mask
            
            # 2. Apply height filter
            min_height = self.params['height_threshold']['min_height']
            max_height = self.params['height_threshold']['max_height']
            
            height_mask = (
                (chunk[z_col] >= min_height) & (chunk[z_col] <= max_height)
            ) | ~valid_mask
            
            # Combine masks
            combined_mask = boundary_mask & height_mask
            
            # Apply masks
            chunk = chunk[combined_mask].reset_index(drop=True)
        
        return chunk
    
    def generate_visualization(self, cleaned_data, save_path):
        """Generate a simple visualization of the cleaned data"""
        plt.figure(figsize=(10, 8))
        
        # Plot boundary
        boundary_x = [
            self.params['boundary']['x_min'], self.params['boundary']['x_max'],
            self.params['boundary']['x_max'], self.params['boundary']['x_min'],
            self.params['boundary']['x_min']
        ]
        boundary_y = [
            self.params['boundary']['y_min'], self.params['boundary']['y_min'],
            self.params['boundary']['y_max'], self.params['boundary']['y_max'],
            self.params['boundary']['y_min']
        ]
        plt.plot(boundary_x, boundary_y, 'k-', linewidth=2, label='Boundary')
        
        # Plot robot 1 points
        mask1 = ~cleaned_data[['robot_1_global_x_radar', 'robot_1_global_y_radar']].isna().any(axis=1)
        if mask1.any():
            plt.scatter(
                cleaned_data.loc[mask1, 'robot_1_global_x_radar'],
                cleaned_data.loc[mask1, 'robot_1_global_y_radar'],
                color='blue', alpha=0.5, s=2.0, label='Robot 1'
            )
        
        # Plot robot 2 points
        mask2 = ~cleaned_data[['robot_2_global_x_radar', 'robot_2_global_y_radar']].isna().any(axis=1)
        if mask2.any():
            plt.scatter(
                cleaned_data.loc[mask2, 'robot_2_global_x_radar'],
                cleaned_data.loc[mask2, 'robot_2_global_y_radar'],
                color='red', alpha=0.5, s=2.0, label='Robot 2'
            )
        
        plt.legend()
        plt.grid(True)
        plt.title(f"Cleaned Point Cloud - {len(cleaned_data)} points")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization to {save_path}")


def process_folder(input_folder, output_folder=None):
    """Process all transformed data files in a folder structure"""
    if output_folder is None:
        output_folder = os.path.join(os.path.dirname(input_folder), "Cleaned_Data")
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all transformed CSV files
    transformed_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.startswith("transformed_") and file.endswith(".csv"):
                transformed_files.append(os.path.join(root, file))
    
    print(f"Found {len(transformed_files)} transformed data files")
    
    # Process each file
    for i, input_path in enumerate(transformed_files):
        print(f"\n[{i+1}/{len(transformed_files)}] Processing: {input_path}")
        
        # Determine relative path from input_folder
        rel_path = os.path.relpath(os.path.dirname(input_path), input_folder)
        
        # Create corresponding output directory
        output_dir = os.path.join(output_folder, rel_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Output path for cleaned data
        output_path = os.path.join(output_dir, os.path.basename(input_path).replace("transformed_", "cleaned_"))
        
        # Skip if already processed
        if os.path.exists(output_path):
            print(f"Output file already exists, skipping: {output_path}")
            continue
        
        # Process file
        try:
            cleaner = MemoryEfficientCleaner(input_path, output_path)
            cleaner.process()
            
            # Force garbage collection
            gc.collect()
        except Exception as e:
            print(f"Error processing file: {e}")


def main():
    parser = argparse.ArgumentParser(description='Clean point cloud data with memory efficiency')
    parser.add_argument('--input', '-i', type=str, 
                        help='Input file or directory')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output file or directory')
    parser.add_argument('--batch', '-b', action='store_true',
                        help='Process entire directory structure')
    
    args = parser.parse_args()
    
    if args.batch:
        if not args.input:
            args.input = '/media/yugi/MAIN DRIVE/RoboFUSE_Dataset/Transformed_Data'
        
        process_folder(args.input, args.output)
    else:
        if not args.input:
            print("Error: Input file required")
            return 1
        
        cleaner = MemoryEfficientCleaner(args.input, args.output)
        cleaner.process()
    
    return 0


if __name__ == "__main__":
    main()
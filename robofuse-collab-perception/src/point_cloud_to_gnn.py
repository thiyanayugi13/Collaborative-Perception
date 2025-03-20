#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import glob
from pathlib import Path
import time
from scipy.spatial import cKDTree
import json

class PointCloudToGNNConverter:
    """
    Converts cleaned point cloud data to GNN-ready format for collaborative perception
    """
    
    def __init__(self, voxel_size=0.3, k_neighbors=8, collaboration_dist=2.0):
        """
        Initialize converter with parameters
        
        Args:
            voxel_size: Size of voxels in meters
            k_neighbors: Number of nearest neighbors for each node
            collaboration_dist: Maximum distance for collaborative edges
        """
        self.voxel_size = voxel_size
        self.k_neighbors = k_neighbors
        self.collaboration_dist = collaboration_dist
        
        # Column mapping for different datasets
        self.column_mapping = {
            'robot_1_global_x_radar': ['robot_1_global_x_radar', 'global_x_radar_robot_1'],
            'robot_1_global_y_radar': ['robot_1_global_y_radar', 'global_y_radar_robot_1'],
            'robot_1_global_z_radar': ['robot_1_global_z_radar', 'global_z_radar_robot_1'],
            'robot_2_global_x_radar': ['robot_2_global_x_radar', 'global_x_radar_robot_2'],
            'robot_2_global_y_radar': ['robot_2_global_y_radar', 'global_y_radar_robot_2'],
            'robot_2_global_z_radar': ['robot_2_global_z_radar', 'global_z_radar_robot_2']
        }
        
    def voxelize_point_cloud(self, points_df):
        """
        Transform raw point cloud into voxel grid representation
        
        Args:
            points_df: DataFrame with cleaned point cloud data
            
        Returns:
            Dictionary mapping voxel coordinates to node information
        """
        voxel_grid = {}
        
        # Find the appropriate column names
        r1x_col = self._find_column(points_df, self.column_mapping['robot_1_global_x_radar'])
        r1y_col = self._find_column(points_df, self.column_mapping['robot_1_global_y_radar'])
        r1z_col = self._find_column(points_df, self.column_mapping['robot_1_global_z_radar'])
        r2x_col = self._find_column(points_df, self.column_mapping['robot_2_global_x_radar'])
        r2y_col = self._find_column(points_df, self.column_mapping['robot_2_global_y_radar'])
        r2z_col = self._find_column(points_df, self.column_mapping['robot_2_global_z_radar'])
        
        # Process robot 1 points
        if r1x_col and r1y_col and r1z_col:
            for idx, row in points_df.iterrows():
                # Skip rows with missing data
                if (pd.isna(row[r1x_col]) or 
                    pd.isna(row[r1y_col]) or 
                    pd.isna(row[r1z_col])):
                    continue
                
                # Calculate voxel indices
                vx = int(row[r1x_col] / self.voxel_size)
                vy = int(row[r1y_col] / self.voxel_size)
                vz = int(row[r1z_col] / self.voxel_size)
                voxel_key = (vx, vy, vz)
                
                # Create or update voxel
                if voxel_key not in voxel_grid:
                    voxel_grid[voxel_key] = {
                        'center': [
                            (vx + 0.5) * self.voxel_size,  # Center X
                            (vy + 0.5) * self.voxel_size,  # Center Y
                            (vz + 0.5) * self.voxel_size   # Center Z
                        ],
                        'point_count': 0,
                        'robot_counts': [0, 0],  # Counts for [robot_1, robot_2]
                        'points': []
                    }
                
                # Update voxel data
                voxel_grid[voxel_key]['point_count'] += 1
                voxel_grid[voxel_key]['robot_counts'][0] += 1
                voxel_grid[voxel_key]['points'].append({
                    'x': row[r1x_col],
                    'y': row[r1y_col],
                    'z': row[r1z_col],
                    'robot_id': 0
                })
        
        # Process robot 2 points
        if r2x_col and r2y_col and r2z_col:
            for idx, row in points_df.iterrows():
                # Skip rows with missing data
                if (pd.isna(row[r2x_col]) or 
                    pd.isna(row[r2y_col]) or 
                    pd.isna(row[r2z_col])):
                    continue
                
                # Calculate voxel indices
                vx = int(row[r2x_col] / self.voxel_size)
                vy = int(row[r2y_col] / self.voxel_size)
                vz = int(row[r2z_col] / self.voxel_size)
                voxel_key = (vx, vy, vz)
                
                # Create or update voxel
                if voxel_key not in voxel_grid:
                    voxel_grid[voxel_key] = {
                        'center': [
                            (vx + 0.5) * self.voxel_size,
                            (vy + 0.5) * self.voxel_size,
                            (vz + 0.5) * self.voxel_size
                        ],
                        'point_count': 0,
                        'robot_counts': [0, 0],
                        'points': []
                    }
                
                # Update voxel data
                voxel_grid[voxel_key]['point_count'] += 1
                voxel_grid[voxel_key]['robot_counts'][1] += 1
                voxel_grid[voxel_key]['points'].append({
                    'x': row[r2x_col],
                    'y': row[r2y_col],
                    'z': row[r2z_col],
                    'robot_id': 1
                })
        
        return voxel_grid
    
    def _find_column(self, df, possible_names):
        """Find a column in dataframe from a list of possible names"""
        for name in possible_names:
            if name in df.columns:
                return name
        return None
    
    def create_node_features(self, voxel_grid):
        """
        Extract node features from voxel grid for GNN input
        
        Returns:
            node_features: numpy array with shape [num_nodes, num_features]
            node_indices: mapping from voxel keys to node indices
        """
        node_features = []
        node_indices = {}
        
        for i, (voxel_key, voxel_data) in enumerate(voxel_grid.items()):
            # Store mapping from voxel key to node index
            node_indices[voxel_key] = i
            
            # Calculate features
            center = voxel_data['center']
            point_count = voxel_data['point_count']
            robot1_count = voxel_data['robot_counts'][0]
            robot2_count = voxel_data['robot_counts'][1]
            
            # Calculate derived features
            occupancy = min(1.0, point_count / 10.0)  # Normalize with cap at 1.0
            robot1_ratio = robot1_count / point_count if point_count > 0 else 0
            seen_by_both = float((robot1_count > 0) and (robot2_count > 0))
            height_from_ground = center[2]
            observation_balance = 1.0 - abs(robot1_ratio - 0.5) * 2.0  # 1.0 if equal, 0.0 if all from one robot
            
            # Compile feature vector
            features = [
                center[0],               # X position
                center[1],               # Y position
                center[2],               # Z position
                occupancy,               # Occupancy measure
                robot1_ratio,            # Ratio of points from robot_1
                seen_by_both,            # Binary indicator if seen by both robots
                observation_balance,     # Balance of observations from both robots
                height_from_ground       # Height from ground plane
            ]
            
            node_features.append(features)
        
        return np.array(node_features), node_indices
    
    def create_edge_index(self, voxel_grid, node_indices):
        """
        Create edge connections for GNN
        
        Args:
            voxel_grid: Dictionary of voxel data
            node_indices: Mapping from voxel keys to node indices
            
        Returns:
            edge_index: numpy array with shape [2, num_edges]
        """
        # Extract node positions for KNN
        positions = []
        node_ids = []
        
        for voxel_key, idx in node_indices.items():
            center = voxel_grid[voxel_key]['center']
            positions.append(center)
            node_ids.append(idx)
        
        # Calculate K nearest neighbors
        positions = np.array(positions)
        node_ids = np.array(node_ids)
        
        k = min(self.k_neighbors + 1, len(positions))  # +1 because first match is self
        tree = cKDTree(positions)
        distances, indices = tree.query(positions, k=k)
        
        # Create edge list
        edges = []
        for i, neighbors in enumerate(indices):
            source_node = node_ids[i]
            for j in neighbors[1:]:  # Skip first (self)
                target_node = node_ids[j]
                edges.append([source_node, target_node])
                # Add reverse edge for undirected graph
                edges.append([target_node, source_node])
        
        # Convert to proper format [2, num_edges]
        edge_index = np.array(edges).T
        
        return edge_index
    
    def add_collaborative_edges(self, voxel_grid, node_indices, edge_index):
        """
        Add special edges between voxels observed by different robots
        
        Args:
            voxel_grid: Dictionary of voxel data
            node_indices: Mapping from voxel keys to node indices
            edge_index: Existing edge index
            
        Returns:
            Updated edge_index
        """
        # Group nodes by primary robot observer
        robot1_nodes = []
        robot2_nodes = []
        
        for voxel_key, idx in node_indices.items():
            voxel = voxel_grid[voxel_key]
            robot_counts = voxel['robot_counts']
            
            # Add to appropriate group based on which robot observed it more
            if robot_counts[0] > robot_counts[1]:
                robot1_nodes.append((idx, voxel['center']))
            elif robot_counts[1] > robot_counts[0]:
                robot2_nodes.append((idx, voxel['center']))
        
        # Add edges between close nodes from different robots
        collab_edges = []
        
        for idx1, pos1 in robot1_nodes:
            for idx2, pos2 in robot2_nodes:
                # Calculate Euclidean distance
                dist = np.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(pos1, pos2)))
                
                if dist < self.collaboration_dist:
                    collab_edges.append([idx1, idx2])
                    collab_edges.append([idx2, idx1])  # Add reverse edge
        
        # Combine with existing edges
        if len(collab_edges) > 0:
            collab_edge_index = np.array(collab_edges).T
            edge_index = np.concatenate([edge_index, collab_edge_index], axis=1)
        
        return edge_index
    
    def convert_to_pyg_data(self, node_features, edge_index):
        """
        Convert prepared data to PyTorch Geometric Data object
        
        Args:
            node_features: numpy array with shape [num_nodes, num_features]
            edge_index: numpy array with shape [2, num_edges]
            
        Returns:
            PyTorch Geometric Data object
        """
        # Convert to PyTorch tensors
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        
        # Create PyG Data object
        data = Data(x=x, edge_index=edge_index)
        
        return data
    
    def process_file(self, file_path):
        """
        Process a single cleaned point cloud file
        
        Args:
            file_path: Path to CSV file with cleaned point cloud data
            
        Returns:
            PyTorch Geometric Data object, stats dictionary
        """
        print(f"Processing file: {file_path}")
        
        try:
            # Load data
            df = pd.read_csv(file_path)
            
            # Voxelize point cloud
            voxel_grid = self.voxelize_point_cloud(df)
            
            if len(voxel_grid) == 0:
                print(f"Warning: No valid voxels created from {file_path}")
                return None, {'error': 'No valid voxels created'}
            
            # Create node features
            node_features, node_indices = self.create_node_features(voxel_grid)
            
            # Create edge connections
            edge_index = self.create_edge_index(voxel_grid, node_indices)
            
            # Add collaborative edges
            edge_index = self.add_collaborative_edges(voxel_grid, node_indices, edge_index)
            
            # Convert to PyG format
            data = self.convert_to_pyg_data(node_features, edge_index)
            
            # Calculate statistics
            stats = {
                'num_voxels': len(voxel_grid),
                'num_nodes': data.num_nodes,
                'num_edges': data.num_edges,
                'edge_density': data.num_edges / (data.num_nodes ** 2) if data.num_nodes > 0 else 0,
                'voxel_size': self.voxel_size,
                'k_neighbors': self.k_neighbors,
                'collaboration_dist': self.collaboration_dist
            }
            
            print(f"Created graph with {stats['num_nodes']} nodes and {stats['num_edges']} edges")
            return data, stats
            
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            return None, {'error': str(e)}

    def save_processed_data(self, data, output_path):
        """
        Save processed GNN data
        
        Args:
            data: PyTorch Geometric Data object
            output_path: Path to save the data
            
        Returns:
            bool: Success or failure
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save using PyTorch
            torch.save(data, output_path)
            print(f"Saved processed data to {output_path}")
            return True
        except Exception as e:
            print(f"Error saving data to {output_path}: {str(e)}")
            return False


def process_all_datasets(base_dir, output_dir, voxel_size=0.3, k_neighbors=8, collab_dist=2.0):
    """
    Process all cleaned point cloud datasets
    
    Args:
        base_dir: Base directory containing cleaned data
        output_dir: Directory to save processed GNN data
        voxel_size: Voxel size in meters
        k_neighbors: Number of nearest neighbors for each node
        collab_dist: Maximum distance for collaborative edges
    """
    # Initialize converter
    converter = PointCloudToGNNConverter(
        voxel_size=voxel_size,
        k_neighbors=k_neighbors,
        collaboration_dist=collab_dist
    )
    
    # Find all cleaned data files
    cleaned_files = glob.glob(os.path.join(base_dir, '**', 'cleaned_*.csv'), recursive=True)
    
    if not cleaned_files:
        print(f"No cleaned data files found in {base_dir}")
        return
    
    print(f"Found {len(cleaned_files)} cleaned data files")
    
    # Process each file
    overall_stats = {
        'total_files': len(cleaned_files),
        'successful_conversions': 0,
        'failed_conversions': 0,
        'total_nodes': 0,
        'total_edges': 0,
        'processing_time': 0,
        'dataset_stats': {}
    }
    
    start_time = time.time()
    
    for file_path in cleaned_files:
        # Generate output path
        rel_path = os.path.relpath(file_path, base_dir)
        scenario = rel_path.split(os.sep)[0] if os.sep in rel_path else ''
        
        output_file = os.path.join(
            output_dir,
            scenario,
            os.path.basename(file_path).replace('.csv', '_gnn.pt')
        )
        
        # Skip if output file already exists
        if os.path.exists(output_file):
            print(f"Skipping {file_path} (output already exists)")
            continue
        
        # Process file
        data, stats = converter.process_file(file_path)
        
        if data is not None:
            # Save processed data
            success = converter.save_processed_data(data, output_file)
            
            if success:
                overall_stats['successful_conversions'] += 1
                overall_stats['total_nodes'] += stats['num_nodes']
                overall_stats['total_edges'] += stats['num_edges']
                
                # Save scenario-specific stats
                if scenario not in overall_stats['dataset_stats']:
                    overall_stats['dataset_stats'][scenario] = []
                
                file_stats = {
                    'file': os.path.basename(file_path),
                    'nodes': stats['num_nodes'],
                    'edges': stats['num_edges'],
                    'edge_density': stats['edge_density']
                }
                overall_stats['dataset_stats'][scenario].append(file_stats)
            else:
                overall_stats['failed_conversions'] += 1
        else:
            overall_stats['failed_conversions'] += 1
    
    overall_stats['processing_time'] = time.time() - start_time
    
    # Save overall statistics
    stats_file = os.path.join(output_dir, 'processing_stats.json')
    try:
        with open(stats_file, 'w') as f:
            json.dump(overall_stats, f, indent=2)
        print(f"Saved processing statistics to {stats_file}")
    except Exception as e:
        print(f"Error saving statistics: {str(e)}")
    
    print("\nProcessing Summary:")
    print(f"Total files processed: {overall_stats['total_files']}")
    print(f"Successful conversions: {overall_stats['successful_conversions']}")
    print(f"Failed conversions: {overall_stats['failed_conversions']}")
    print(f"Total nodes created: {overall_stats['total_nodes']}")
    print(f"Total edges created: {overall_stats['total_edges']}")
    print(f"Total processing time: {overall_stats['processing_time']:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert cleaned point cloud data to GNN-ready format"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/media/yugi/MAIN DRIVE/RoboFUSE_Dataset/Cleaned_Data",
        help="Base directory containing cleaned data (default: /media/yugi/MAIN DRIVE/RoboFUSE_Dataset/Cleaned_Data)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/media/yugi/MAIN DRIVE/RoboFUSE_Dataset/GNN_Ready_Data",
        help="Directory to save processed GNN data"
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.3,
        help="Voxel size in meters (default: 0.3)"
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=8,
        help="Number of nearest neighbors for each node (default: 8)"
    )
    parser.add_argument(
        "--collab-dist",
        type=float,
        default=2.0,
        help="Maximum distance for collaborative edges (default: 2.0)"
    )
    parser.add_argument(
        "--single-file",
        type=str,
        default=None,
        help="Process only a single file (optional)"
    )
    
    args = parser.parse_args()
    
    if args.single_file:
        # Process single file
        converter = PointCloudToGNNConverter(
            voxel_size=args.voxel_size,
            k_neighbors=args.k_neighbors,
            collaboration_dist=args.collab_dist
        )
        
        output_file = os.path.join(
            args.output_dir,
            os.path.basename(args.single_file).replace('.csv', '_gnn.pt')
        )
        
        data, stats = converter.process_file(args.single_file)
        if data is not None:
            converter.save_processed_data(data, output_file)
    else:
        # Process all datasets
        process_all_datasets(
            args.base_dir,
            args.output_dir,
            args.voxel_size,
            args.k_neighbors,
            args.collab_dist
        )
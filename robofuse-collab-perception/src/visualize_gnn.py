#!/usr/bin/env python3

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
import networkx as nx
import argparse
import os

def visualize_gnn_data(data_path, min_occupancy=0.1, min_height=0.05, max_height=2.5):
    """
    Visualize GNN data with filtering for outliers
    
    Args:
        data_path: Path to PyTorch Geometric data file
        min_occupancy: Minimum occupancy threshold (0-1) to include nodes
        min_height: Minimum height from ground to include
        max_height: Maximum height from ground to include
    """
    print(f"Loading GNN data from: {data_path}")
    
    try:
        # First try loading with weights_only=False for PyTorch 2.6+
        try:
            data = torch.load(data_path, weights_only=False)
        except TypeError:
            # Fall back to standard loading for older PyTorch versions
            data = torch.load(data_path)
        
        # Print basic statistics
        print(f"Number of nodes: {data.num_nodes}")
        print(f"Number of edges: {data.num_edges}")
        print(f"Node features shape: {data.x.shape}")
        print(f"Edge index shape: {data.edge_index.shape}")
        
        # Extract features we care about
        x_pos = data.x[:, 0].numpy()  # X position
        y_pos = data.x[:, 1].numpy()  # Y position
        z_pos = data.x[:, 2].numpy()  # Z position
        occupancy = data.x[:, 3].numpy()  # Occupancy
        robot1_ratio = data.x[:, 4].numpy()  # Ratio from robot 1
        seen_by_both = data.x[:, 5].numpy()  # Seen by both robots
        height = data.x[:, 7].numpy()  # Height from ground
        
        # Create filters for outlier removal
        occupancy_filter = occupancy >= min_occupancy
        height_filter = (height >= min_height) & (height <= max_height)
        
        # Combined filter
        node_filter = occupancy_filter & height_filter
        
        # Count filtered nodes
        original_nodes = data.num_nodes
        filtered_nodes = np.sum(node_filter)
        print(f"Filtered from {original_nodes} to {filtered_nodes} nodes ({filtered_nodes/original_nodes*100:.1f}%)")
        
        # Create a filtered version of the data
        filtered_x = data.x[node_filter]
        
        # Get indices of the filtered nodes
        filtered_indices = np.where(node_filter)[0]
        
        # Create a mapping from original to new indices
        index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(filtered_indices)}
        
        # Filter edges to only include filtered nodes
        filtered_edges = []
        for i in range(data.edge_index.shape[1]):
            src, dst = data.edge_index[0, i].item(), data.edge_index[1, i].item()
            if src in index_mapping and dst in index_mapping:
                filtered_edges.append([index_mapping[src], index_mapping[dst]])
        
        # Create filtered edge index tensor
        if filtered_edges:
            filtered_edge_index = torch.tensor(filtered_edges).t().to(torch.long)
        else:
            filtered_edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        # Print filtered statistics
        print(f"Filtered edges: {filtered_edge_index.shape[1]}")
        
        # Create a new PyG data object with filtered data
        filtered_data = type(data)(
            x=filtered_x,
            edge_index=filtered_edge_index
        )
        
        # Ensure output directories exist
        output_dir = os.path.dirname(data_path)
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        base_filename = os.path.splitext(os.path.basename(data_path))[0]
        
        # Create 2D visualization - Top view
        plt.figure(figsize=(12, 10))
        
        # Extract positions for filtered nodes
        x_filtered = filtered_x[:, 0].numpy()
        y_filtered = filtered_x[:, 1].numpy()
        z_filtered = filtered_x[:, 2].numpy()
        occupancy_filtered = filtered_x[:, 3].numpy()
        robot1_ratio_filtered = filtered_x[:, 4].numpy()
        
        # Create networkx graph from filtered data
        G = to_networkx(filtered_data, to_undirected=True)
        
        # Create position dictionary for nodes
        pos = {i: (x_filtered[i], y_filtered[i]) for i in range(len(x_filtered))}
        
        # Draw the graph - size based on occupancy, color based on robot ratio
        node_size = occupancy_filtered * 100 + 10  # Scale occupancy for visual size
        plt.scatter(x_filtered, y_filtered, c=robot1_ratio_filtered, cmap='coolwarm', 
                    s=node_size, alpha=0.7, edgecolors='black', linewidths=0.5)
        
        # Draw edges from networkx graph
        nx.draw_networkx_edges(G, pos, alpha=0.15, width=0.5)
        
        # Add warehouse boundaries
        warehouse_x = [-9.1, 10.2, 10.2, -9.1, -9.1]
        warehouse_y = [-4.42, -4.42, 5.5, 5.5, -4.42]
        plt.plot(warehouse_x, warehouse_y, 'r--', linewidth=2, label='Arena Boundaries')
        
        # Add color bar for robot ratio
        cbar = plt.colorbar()
        cbar.set_label('Robot 1 Observation Ratio')
        
        # Add labels and title
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title(f'Collaborative Perception Graph (Top View)')
        plt.grid(True, alpha=0.3)
        
        # Add stats as text
        stats_text = (
            f"Nodes: {filtered_nodes}/{original_nodes}\n"
            f"Edges: {filtered_edge_index.shape[1]}\n"
            f"Min occupancy: {min_occupancy}\n"
            f"Height range: {min_height}-{max_height}m"
        )
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                bbox=dict(facecolor='white', alpha=0.7), 
                verticalalignment='top', fontsize=9)
        
        # Show plot with equal axis aspect ratio
        plt.axis('equal')
        
        # Save 2D visualization
        output_file_2d = os.path.join(vis_dir, f"{base_filename}_2d.png")
        plt.savefig(output_file_2d, dpi=300, bbox_inches='tight')
        print(f"Saved 2D visualization to: {output_file_2d}")
        plt.close()
        
        # Create occupancy histogram
        plt.figure(figsize=(10, 6))
        plt.hist(occupancy, bins=20, alpha=0.7, label='Occupancy')
        plt.axvline(min_occupancy, color='r', linestyle='--', label=f'Min Threshold ({min_occupancy})')
        plt.xlabel('Occupancy Value')
        plt.ylabel('Count')
        plt.title('Distribution of Occupancy Values')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save occupancy histogram
        output_file_hist = os.path.join(vis_dir, f"{base_filename}_occupancy_hist.png")
        plt.savefig(output_file_hist, dpi=300)
        plt.close()
        
        # Create 3D visualization
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot 3D scatter with colored points by robot ratio
        scatter = ax.scatter(x_filtered, y_filtered, z_filtered, 
                             c=robot1_ratio_filtered, cmap='coolwarm', 
                             s=node_size,
                             alpha=0.7, edgecolors='black', linewidths=0.5)
        
        # Draw a subset of edges to avoid cluttering the visualization
        max_edges = min(5000, filtered_edge_index.shape[1])
        edge_indices = np.random.choice(filtered_edge_index.shape[1], max_edges, replace=False) if filtered_edge_index.shape[1] > 0 else []
        
        for i in edge_indices:
            src, dst = filtered_edge_index[0, i].item(), filtered_edge_index[1, i].item()
            ax.plot([x_filtered[src], x_filtered[dst]], 
                    [y_filtered[src], y_filtered[dst]], 
                    [z_filtered[src], z_filtered[dst]], 
                    'gray', alpha=0.15, linewidth=0.5)
        
        # Add warehouse boundaries - base of the arena
        ax.plot(warehouse_x, warehouse_y, [0]*len(warehouse_x), 'r--', linewidth=2)
        
        # Add color bar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Robot 1 Observation Ratio')
        
        # Add labels
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Z Position (m)')
        ax.set_title('3D Collaborative Perception')
        
        # Add stats as text
        ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes, 
                  bbox=dict(facecolor='white', alpha=0.7), 
                  verticalalignment='top', fontsize=9)
        
        # Save 3D visualization
        output_file_3d = os.path.join(vis_dir, f"{base_filename}_3d.png")
        plt.savefig(output_file_3d, dpi=300)
        plt.close()
        
        # Create robot ratio distribution plot
        plt.figure(figsize=(10, 6))
        plt.hist(robot1_ratio, bins=20, alpha=0.7, color='purple')
        plt.axvline(0.5, color='k', linestyle='--', label='Equal contribution')
        plt.xlabel('Robot 1 Observation Ratio')
        plt.ylabel('Count')
        plt.title('Distribution of Robot Contributions')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        output_file_ratio = os.path.join(vis_dir, f"{base_filename}_robot_ratio.png")
        plt.savefig(output_file_ratio, dpi=300)
        plt.close()
        
        # Calculate and save statistics
        collab_nodes = np.sum((robot1_ratio_filtered > 0.2) & (robot1_ratio_filtered < 0.8))
        collab_percentage = (collab_nodes / len(robot1_ratio_filtered)) * 100 if len(robot1_ratio_filtered) > 0 else 0
        
        stats = {
            "nodes": {
                "total": int(original_nodes),
                "filtered": int(filtered_nodes),
                "percentage_kept": float(f"{filtered_nodes/original_nodes*100:.1f}")
            },
            "edges": {
                "total": int(filtered_edge_index.shape[1])
            },
            "collaboration": {
                "collaborative_nodes": int(collab_nodes),
                "collaborative_percentage": float(f"{collab_percentage:.1f}"),
                "robot1_dominant": int(np.sum(robot1_ratio_filtered > 0.8)),
                "robot2_dominant": int(np.sum(robot1_ratio_filtered < 0.2))
            },
            "coverage": {
                "x_range": [float(f"{x_filtered.min():.2f}"), float(f"{x_filtered.max():.2f}")],
                "y_range": [float(f"{y_filtered.min():.2f}"), float(f"{y_filtered.max():.2f}")],
                "z_range": [float(f"{z_filtered.min():.2f}"), float(f"{z_filtered.max():.2f}")]
            },
            "filter_params": {
                "min_occupancy": float(min_occupancy),
                "min_height": float(min_height),
                "max_height": float(max_height)
            }
        }
        
        # Save statistics to text file
        stats_str = (
            f"GNN Data Statistics\n"
            f"==================\n\n"
            f"Nodes:\n"
            f"  Total: {stats['nodes']['total']}\n"
            f"  Filtered: {stats['nodes']['filtered']} ({stats['nodes']['percentage_kept']}%)\n"
            f"Edges:\n"
            f"  Total: {stats['edges']['total']}\n"
            f"Collaboration:\n"
            f"  Collaborative nodes: {stats['collaboration']['collaborative_nodes']} ({stats['collaboration']['collaborative_percentage']}%)\n"
            f"  Robot 1 dominant: {stats['collaboration']['robot1_dominant']}\n"
            f"  Robot 2 dominant: {stats['collaboration']['robot2_dominant']}\n"
            f"Coverage:\n"
            f"  X range: {stats['coverage']['x_range'][0]} to {stats['coverage']['x_range'][1]} m\n"
            f"  Y range: {stats['coverage']['y_range'][0]} to {stats['coverage']['y_range'][1]} m\n"
            f"  Z range: {stats['coverage']['z_range'][0]} to {stats['coverage']['z_range'][1]} m\n"
            f"Filter Parameters:\n"
            f"  Min occupancy: {stats['filter_params']['min_occupancy']}\n"
            f"  Min height: {stats['filter_params']['min_height']} m\n"
            f"  Max height: {stats['filter_params']['max_height']} m\n"
        )
        
        stats_file = os.path.join(vis_dir, f"{base_filename}_stats.txt")
        with open(stats_file, 'w') as f:
            f.write(stats_str)
        
        print(f"Saved statistics to: {stats_file}")
        print(f"All visualizations saved to: {vis_dir}")
        
        return True
    
    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize GNN data with outlier filtering")
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to PyTorch Geometric data file (.pt)')
    parser.add_argument('--min-occupancy', type=float, default=0.2,
                        help='Minimum occupancy threshold (0-1) to include nodes')
    parser.add_argument('--min-height', type=float, default=0.05,
                        help='Minimum height from ground to include')
    parser.add_argument('--max-height', type=float, default=2.0,
                        help='Maximum height from ground to include')
    
    args = parser.parse_args()
    
    visualize_gnn_data(args.input, args.min_occupancy, args.min_height, args.max_height)
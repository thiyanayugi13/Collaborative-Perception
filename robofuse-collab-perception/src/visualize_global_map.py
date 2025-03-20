#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os
from pathlib import Path
from matplotlib.animation import FuncAnimation


class GlobalMapVisualizer:
    """
    A class to visualize the global point cloud map created from
    collaborative perception of multiple robots.
    """
    
    def __init__(self, data_path=None):
        """
        Initialize the visualizer with the path to transformed data.
        
        Args:
            data_path (str): Path to the transformed data CSV file
        """
        self.data_path = data_path
        self.data = None
        
        # Warehouse boundaries (can be adjusted based on your environment)
        self.boundaries = {
            'x_min': -9.1, 'x_max': 10.2,
            'y_min': -4.42, 'y_max': 5.5,
            'z_min': 0.0, 'z_max': 3.0
        }
        
        # Store workstation positions (if available)
        self.workstations = {
            'AS_1_neu': {'x': 1.52, 'y': 2.24, 'z': 1.02},
            'AS_3_neu': {'x': -5.74, 'y': -0.13, 'z': 1.47},
            'AS_4_neu': {'x': 5.37, 'y': 0.21, 'z': 2.30},
            'AS_5_neu': {'x': -3.05, 'y': 2.39, 'z': 2.21},
            'AS_6_neu': {'x': 0.01, 'y': -1.45, 'z': 1.53}
        }
        
        # Workstation dimensions
        self.workstation_size = {'width': 1.0, 'height': 0.6, 'depth': 0.8}
        
        # Color mapping for visualization
        self.colors = {
            'robot_1': 'blue',
            'robot_2': 'red',
            'workstation': 'gray',
            'boundary': 'black'
        }
        
    def load_data(self, data_path=None):
        """
        Load transformed point cloud data from CSV file.
        
        Args:
            data_path (str, optional): Path to override the initialized path
            
        Returns:
            bool: True if data was loaded successfully, False otherwise
        """
        if data_path:
            self.data_path = data_path
            
        if not self.data_path:
            print("Error: No data path provided")
            return False
        
        try:
            print(f"Loading transformed data from: {self.data_path}")
            self.data = pd.read_csv(self.data_path)
            
            # Check if it's a global point cloud or a transformed dataset
            if 'global_x' in self.data.columns:
                # Already in global point cloud format
                pass
            elif 'robot_1_global_x_radar' in self.data.columns:
                # Convert from transformer output to global point cloud
                self.data = self.create_global_point_cloud()
            else:
                print("Error: Data does not appear to be in the expected format")
                return False
                
            print(f"Loaded {len(self.data)} data points")
            print(f"Available columns: {', '.join(self.data.columns)}")
            
            # Store unique timestamps for possible animation
            self.timestamps = self.data['vicon_timestamp'].unique()
            self.timestamps.sort()
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
    
    def create_global_point_cloud(self):
        """
        Convert from transformer output format to global point cloud format.
        
        Returns:
            pandas.DataFrame: Reformatted global point cloud
        """
        # Extract global radar points from each robot
        robot1_points = self.data[[
            'vicon_timestamp',
            'robot_1_global_x_radar', 
            'robot_1_global_y_radar', 
            'robot_1_global_z_radar'
        ]].rename(columns={
            'robot_1_global_x_radar': 'global_x',
            'robot_1_global_y_radar': 'global_y',
            'robot_1_global_z_radar': 'global_z'
        })
        robot1_points['source'] = 'robot_1'
        
        robot2_points = self.data[[
            'vicon_timestamp',
            'robot_2_global_x_radar', 
            'robot_2_global_y_radar', 
            'robot_2_global_z_radar'
        ]].rename(columns={
            'robot_2_global_x_radar': 'global_x',
            'robot_2_global_y_radar': 'global_y',
            'robot_2_global_z_radar': 'global_z'
        })
        robot2_points['source'] = 'robot_2'
        
        # Combine points
        global_point_cloud = pd.concat([robot1_points, robot2_points])
        
        # Remove NaN values
        global_point_cloud = global_point_cloud.dropna(subset=['global_x', 'global_y', 'global_z'])
        
        print(f"Created global point cloud with {len(global_point_cloud)} points")
        return global_point_cloud
    
    def plot_static_map(self, plot_type='2d'):
        """
        Create a static visualization of the global point cloud.
        
        Args:
            plot_type (str): Type of plot ('2d' or '3d')
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        if self.data is None:
            print("Error: No data loaded")
            return None
            
        if plot_type == '3d':
            # Create 3D plot
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot points by source
            for source in self.data['source'].unique():
                subset = self.data[self.data['source'] == source]
                ax.scatter(subset['global_x'], subset['global_y'], subset['global_z'],
                          color=self.colors.get(source, 'gray'),
                          alpha=0.7, s=10, label=f'Points from {source}')
            
            # Add workstations
            for name, pos in self.workstations.items():
                ax.scatter(pos['x'], pos['y'], pos['z'], 
                          color='black', marker='s', s=100, label=name if name=='AS_1_neu' else "")
                ax.text(pos['x'], pos['y'], pos['z'] + 0.2, name, 
                       horizontalalignment='center', size=8)
            
            # Set axis limits
            ax.set_xlim([self.boundaries['x_min'], self.boundaries['x_max']])
            ax.set_ylim([self.boundaries['y_min'], self.boundaries['y_max']])
            ax.set_zlim([self.boundaries['z_min'], self.boundaries['z_max']])
            
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            
        else:
            # Create 2D plot (top-down view)
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Plot warehouse boundary
            boundary_x = [
                self.boundaries['x_min'], self.boundaries['x_max'],
                self.boundaries['x_max'], self.boundaries['x_min'],
                self.boundaries['x_min']
            ]
            boundary_y = [
                self.boundaries['y_min'], self.boundaries['y_min'],
                self.boundaries['y_max'], self.boundaries['y_max'],
                self.boundaries['y_min']
            ]
            ax.plot(boundary_x, boundary_y, 'k-', linewidth=2, label='Warehouse Boundary')
            
            # Plot points by source
            for source in self.data['source'].unique():
                subset = self.data[self.data['source'] == source]
                ax.scatter(subset['global_x'], subset['global_y'],
                          color=self.colors.get(source, 'gray'),
                          alpha=0.7, s=10, label=f'Points from {source}')
            
            # Add workstations (top-down view rectangles)
            for name, pos in self.workstations.items():
                width = self.workstation_size['width']
                height = self.workstation_size['height']
                
                # Special case for AS_4 which is rotated 90 degrees
                if name == 'AS_4_neu':
                    width, height = height, width
                
                rect = plt.Rectangle(
                    (pos['x'] - width/2, pos['y'] - height/2),
                    width, height,
                    linewidth=1, edgecolor='black', facecolor='lightgray',
                    alpha=0.7, zorder=2
                )
                ax.add_patch(rect)
                ax.text(pos['x'], pos['y'], name, 
                       horizontalalignment='center', verticalalignment='center',
                       size=8, zorder=3)
            
            # Set axis limits
            ax.set_xlim([self.boundaries['x_min'] - 0.5, self.boundaries['x_max'] + 0.5])
            ax.set_ylim([self.boundaries['y_min'] - 0.5, self.boundaries['y_max'] + 0.5])
            
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_aspect('equal')
            
        # Add legend and title
        plt.title('Global Point Cloud Map from Collaborative Perception')
        plt.legend(loc='upper right')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        return fig
    
    def create_density_map(self, resolution=0.2):
        """
        Create a 2D density map of the point cloud.
        
        Args:
            resolution (float): Grid cell size in meters
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        if self.data is None:
            print("Error: No data loaded")
            return None
        
        # Create 2D histogram (density map)
        x_bins = np.arange(self.boundaries['x_min'], self.boundaries['x_max'] + resolution, resolution)
        y_bins = np.arange(self.boundaries['y_min'], self.boundaries['y_max'] + resolution, resolution)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot histogram
        hist, x_edges, y_edges = np.histogram2d(
            self.data['global_x'], 
            self.data['global_y'], 
            bins=[x_bins, y_bins]
        )
        
        # Transpose to match the correct orientation
        hist = hist.T
        
        # Plot as heatmap
        im = ax.imshow(
            hist, 
            interpolation='bilinear', 
            origin='lower',
            extent=[self.boundaries['x_min'], self.boundaries['x_max'], 
                   self.boundaries['y_min'], self.boundaries['y_max']],
            aspect='auto',
            cmap='viridis'
        )
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Point Density')
        
        # Add workstations
        for name, pos in self.workstations.items():
            width = self.workstation_size['width']
            height = self.workstation_size['height']
            
            # Special case for AS_4 which is rotated 90 degrees
            if name == 'AS_4_neu':
                width, height = height, width
            
            rect = plt.Rectangle(
                (pos['x'] - width/2, pos['y'] - height/2),
                width, height,
                linewidth=1, edgecolor='black', facecolor='none',
                zorder=2
            )
            ax.add_patch(rect)
            ax.text(pos['x'], pos['y'], name, 
                   color='white', fontweight='bold',
                   horizontalalignment='center', verticalalignment='center',
                   size=8, zorder=3)
        
        # Add warehouse boundary
        boundary_x = [
            self.boundaries['x_min'], self.boundaries['x_max'],
            self.boundaries['x_max'], self.boundaries['x_min'],
            self.boundaries['x_min']
        ]
        boundary_y = [
            self.boundaries['y_min'], self.boundaries['y_min'],
            self.boundaries['y_max'], self.boundaries['y_max'],
            self.boundaries['y_min']
        ]
        ax.plot(boundary_x, boundary_y, 'w-', linewidth=2)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Point Cloud Density Map')
        
        return fig
    
    def animate_point_cloud(self, interval=100, save_path=None):
        """
        Create an animation of the point cloud over time.
        
        Args:
            interval (int): Time between frames in milliseconds
            save_path (str): Path to save the animation (if None, display instead)
            
        Returns:
            matplotlib.animation.FuncAnimation: The animation object
        """
        if self.data is None or 'vicon_timestamp' not in self.data.columns:
            print("Error: Data not loaded or missing timestamp information")
            return None
        
        # Get unique timestamps
        timestamps = np.sort(self.data['vicon_timestamp'].unique())
        if len(timestamps) < 2:
            print("Error: Not enough timesteps for animation")
            return None
        
        # Create figure and initial empty plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot warehouse boundary
        boundary_x = [
            self.boundaries['x_min'], self.boundaries['x_max'],
            self.boundaries['x_max'], self.boundaries['x_min'],
            self.boundaries['x_min']
        ]
        boundary_y = [
            self.boundaries['y_min'], self.boundaries['y_min'],
            self.boundaries['y_max'], self.boundaries['y_max'],
            self.boundaries['y_min']
        ]
        ax.plot(boundary_x, boundary_y, 'k-', linewidth=2)
        
        # Add workstations
        for name, pos in self.workstations.items():
            width = self.workstation_size['width']
            height = self.workstation_size['height']
            
            # Special case for AS_4 which is rotated 90 degrees
            if name == 'AS_4_neu':
                width, height = height, width
            
            rect = plt.Rectangle(
                (pos['x'] - width/2, pos['y'] - height/2),
                width, height,
                linewidth=1, edgecolor='black', facecolor='lightgray',
                alpha=0.7, zorder=2
            )
            ax.add_patch(rect)
            ax.text(pos['x'], pos['y'], name, 
                   horizontalalignment='center', verticalalignment='center',
                   size=8, zorder=3)
        
        # Set plot limits and labels
        ax.set_xlim([self.boundaries['x_min'] - 0.5, self.boundaries['x_max'] + 0.5])
        ax.set_ylim([self.boundaries['y_min'] - 0.5, self.boundaries['y_max'] + 0.5])
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Create empty scatter plots for each robot
        robot1_scatter = ax.scatter([], [], color=self.colors['robot_1'], 
                                   alpha=0.7, s=10, label='Robot 1 Points')
        robot2_scatter = ax.scatter([], [], color=self.colors['robot_2'], 
                                   alpha=0.7, s=10, label='Robot 2 Points')
        
        # Create time text display
        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                           fontsize=10, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.title('Collaborative Perception Point Cloud Over Time')
        plt.legend(loc='upper right')
        
        # Time window for points (in seconds)
        time_window = 1.0  # Show points within this time window
        
        def init():
            """Initialize animation"""
            robot1_scatter.set_offsets(np.empty((0, 2)))
            robot2_scatter.set_offsets(np.empty((0, 2)))
            time_text.set_text('')
            return robot1_scatter, robot2_scatter, time_text
        
        def animate(i):
            """Update animation frame"""
            # Get current timestamp
            current_time = timestamps[i]
            
            # Get data within time window
            window_start = current_time - time_window
            window_data = self.data[
                (self.data['vicon_timestamp'] >= window_start) & 
                (self.data['vicon_timestamp'] <= current_time)
            ]
            
            # Update scatter plots
            robot1_data = window_data[window_data['source'] == 'robot_1']
            robot2_data = window_data[window_data['source'] == 'robot_2']
            
            if len(robot1_data) > 0:
                robot1_scatter.set_offsets(robot1_data[['global_x', 'global_y']].values)
            else:
                robot1_scatter.set_offsets(np.empty((0, 2)))
                
            if len(robot2_data) > 0:
                robot2_scatter.set_offsets(robot2_data[['global_x', 'global_y']].values)
            else:
                robot2_scatter.set_offsets(np.empty((0, 2)))
            
            # Update time text
            rel_time = current_time - timestamps[0]
            time_text.set_text(f'Time: {rel_time:.2f}s')
            
            return robot1_scatter, robot2_scatter, time_text
        
        # Create animation
        anim = FuncAnimation(fig, animate, init_func=init,
                            frames=len(timestamps), interval=interval, 
                            blit=True)
        
        # Save or display animation
        if save_path:
            anim.save(save_path, writer='pillow', fps=30)
            print(f"Animation saved to: {save_path}")
        
        return anim, fig
    
    def analyze_coverage(self, grid_size=0.2):
        """
        Analyze the coverage of the point cloud.
        
        Args:
            grid_size (float): Size of grid cells in meters
            
        Returns:
            dict: Coverage metrics
        """
        if self.data is None:
            print("Error: No data loaded")
            return None
        
        # Create grid
        x_bins = np.arange(self.boundaries['x_min'], self.boundaries['x_max'] + grid_size, grid_size)
        y_bins = np.arange(self.boundaries['y_min'], self.boundaries['y_max'] + grid_size, grid_size)
        
        # Create occupancy grid
        hist, x_edges, y_edges = np.histogram2d(
            self.data['global_x'], 
            self.data['global_y'], 
            bins=[x_bins, y_bins]
        )
        
        # Calculate metrics
        total_cells = len(x_bins) * len(y_bins)
        occupied_cells = np.sum(hist > 0)
        coverage_percentage = 100 * occupied_cells / total_cells
        
        # Calculate coverage by robot
        robot1_data = self.data[self.data['source'] == 'robot_1']
        robot2_data = self.data[self.data['source'] == 'robot_2']
        
        hist_robot1, _, _ = np.histogram2d(
            robot1_data['global_x'], 
            robot1_data['global_y'], 
            bins=[x_bins, y_bins]
        )
        
        hist_robot2, _, _ = np.histogram2d(
            robot2_data['global_x'], 
            robot2_data['global_y'], 
            bins=[x_bins, y_bins]
        )
        
        occupied_robot1 = np.sum(hist_robot1 > 0)
        occupied_robot2 = np.sum(hist_robot2 > 0)
        
        # Calculate overlap
        overlap_cells = np.sum((hist_robot1 > 0) & (hist_robot2 > 0))
        overlap_percentage = 100 * overlap_cells / occupied_cells if occupied_cells > 0 else 0
        
        # Store in metrics dict
        metrics = {
            'grid_size': grid_size,
            'total_cells': total_cells,
            'occupied_cells': int(occupied_cells),
            'coverage_percentage': coverage_percentage,
            'robot1_cells': int(occupied_robot1),
            'robot2_cells': int(occupied_robot2),
            'overlap_cells': int(overlap_cells),
            'overlap_percentage': overlap_percentage
        }
        
        print(f"Coverage Analysis (grid size: {grid_size}m)")
        print(f"  Total area: {total_cells * grid_size * grid_size:.2f} m²")
        print(f"  Covered area: {occupied_cells * grid_size * grid_size:.2f} m²")
        print(f"  Coverage percentage: {coverage_percentage:.2f}%")
        print(f"  Robot 1 coverage: {occupied_robot1 * grid_size * grid_size:.2f} m²")
        print(f"  Robot 2 coverage: {occupied_robot2 * grid_size * grid_size:.2f} m²")
        print(f"  Overlap area: {overlap_cells * grid_size * grid_size:.2f} m²")
        print(f"  Overlap percentage: {overlap_percentage:.2f}%")
        
        return metrics


def main():
    """Main function to run the visualization."""
    parser = argparse.ArgumentParser(description='Visualize global point cloud map')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to transformed data CSV file')
    parser.add_argument('--plot-type', '-p', type=str, choices=['2d', '3d', 'density', 'animate', 'none'], 
                        default='2d', help='Type of visualization (default: 2d)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Path to save visualization (default: display only)')
    parser.add_argument('--analyze', '-a', action='store_true',
                        help='Perform coverage analysis')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = GlobalMapVisualizer()
    
    # Load data
    if not visualizer.load_data(args.input):
        return 1
    
    # Analyze coverage if requested
    if args.analyze:
        visualizer.analyze_coverage()
    
    # Visualization
    if args.plot_type == '2d':
        fig = visualizer.plot_static_map(plot_type='2d')
        
    elif args.plot_type == '3d':
        fig = visualizer.plot_static_map(plot_type='3d')
        
    elif args.plot_type == 'density':
        fig = visualizer.create_density_map()
        
    elif args.plot_type == 'animate':
        anim, fig = visualizer.animate_point_cloud(save_path=args.output)
        if args.output is None:
            plt.show()
        return 0
    
    # Save or display
    if args.output and args.plot_type != 'animate':
        plt.savefig(args.output, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to: {args.output}")
    elif args.plot_type != 'none':
        plt.show()
    
    return 0


if __name__ == "__main__":
    main()
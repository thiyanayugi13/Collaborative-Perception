#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path
from scipy.spatial import cKDTree
from sklearn.metrics import mean_squared_error


class TransformationEvaluator:
    """
    A class to evaluate the accuracy of coordinate transformations
    in the collaborative perception framework.
    """
    
    def __init__(self, data_path=None):
        """
        Initialize the evaluator with the path to transformed data.
        
        Args:
            data_path (str): Path to the transformed data CSV file
        """
        self.data_path = data_path
        self.data = None
        self.evaluation_results = {}
        
    def load_data(self, data_path=None):
        """
        Load transformed data from CSV file.
        
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
            
            # Check if it has the necessary columns for evaluation
            required_cols = [
                'robot_1_global_x', 'robot_1_global_y', 'robot_1_global_z',
                'robot_2_global_x', 'robot_2_global_y', 'robot_2_global_z',
                'robot_1_global_x_radar', 'robot_1_global_y_radar', 'robot_1_global_z_radar',
                'robot_2_global_x_radar', 'robot_2_global_y_radar', 'robot_2_global_z_radar'
            ]
            
            missing_cols = [col for col in required_cols if col not in self.data.columns]
            if missing_cols:
                print(f"Warning: Missing columns for full evaluation: {', '.join(missing_cols)}")
                
            print(f"Loaded {len(self.data)} data points")
            return True
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
        
    def calculate_iou_2d(self, grid_size=0.2):
        """
        Calculate Intersection over Union (IoU) for 2D occupancy grids.
        
        Args:
            grid_size (float): Grid cell size in meters
            
        Returns:
            dict: IoU metrics
        """
        if self.data is None:
            print("Error: No data loaded")
            return None
        
        # Define grid boundaries
        x_min, x_max = -10.0, 11.0
        y_min, y_max = -5.0, 6.0
        
        # Create grids
        x_bins = np.arange(x_min, x_max + grid_size, grid_size)
        y_bins = np.arange(y_min, y_max + grid_size, grid_size)
        
        # Create separate occupancy grids for each robot
        grid_robot1 = np.zeros((len(y_bins)-1, len(x_bins)-1), dtype=bool)
        grid_robot2 = np.zeros((len(y_bins)-1, len(x_bins)-1), dtype=bool)
        
        # Fill grids with transformed radar points
        for _, row in self.data.dropna(subset=['robot_1_global_x_radar', 'robot_1_global_y_radar']).iterrows():
            x, y = row['robot_1_global_x_radar'], row['robot_1_global_y_radar']
            if x_min <= x < x_max and y_min <= y < y_max:
                x_idx = int((x - x_min) / grid_size)
                y_idx = int((y - y_min) / grid_size)
                grid_robot1[y_idx, x_idx] = True
        
        for _, row in self.data.dropna(subset=['robot_2_global_x_radar', 'robot_2_global_y_radar']).iterrows():
            x, y = row['robot_2_global_x_radar'], row['robot_2_global_y_radar']
            if x_min <= x < x_max and y_min <= y < y_max:
                x_idx = int((x - x_min) / grid_size)
                y_idx = int((y - y_min) / grid_size)
                grid_robot2[y_idx, x_idx] = True
        
        # Calculate intersection and union
        intersection = np.logical_and(grid_robot1, grid_robot2).sum()
        union = np.logical_or(grid_robot1, grid_robot2).sum()
        iou = intersection / union if union > 0 else 0
        
        # Store metrics
        metrics = {
            'grid_size': grid_size,
            'robot1_cells': grid_robot1.sum(),
            'robot2_cells': grid_robot2.sum(),
            'intersection_cells': intersection,
            'union_cells': union,
            'iou': iou
        }
        
        print(f"2D IoU Evaluation (grid size: {grid_size}m)")
        print(f"  Robot 1 occupied cells: {grid_robot1.sum()}")
        print(f"  Robot 2 occupied cells: {grid_robot2.sum()}")
        print(f"  Intersection: {intersection} cells")
        print(f"  Union: {union} cells")
        print(f"  IoU: {iou:.4f}")
        
        self.evaluation_results['iou_2d'] = metrics
        return metrics
        
    def calculate_rmse(self):
        """
        Calculate Root Mean Square Error between robot positions and radar detections.
        
        Returns:
            dict: RMSE metrics
        """
        if self.data is None:
            print("Error: No data loaded")
            return None
            
        # Filter out rows with NaN values
        data = self.data.dropna(subset=[
            'robot_1_global_x', 'robot_1_global_y', 'robot_1_global_z',
            'robot_1_global_x_radar', 'robot_1_global_y_radar', 'robot_1_global_z_radar',
            'robot_2_global_x', 'robot_2_global_y', 'robot_2_global_z',
            'robot_2_global_x_radar', 'robot_2_global_y_radar', 'robot_2_global_z_radar'
        ])
        
        if len(data) == 0:
            print("Error: No valid data for RMSE calculation")
            return None
            
        # Calculate distance between robot position and radar-detected positions
        robot1_position = data[['robot_1_global_x', 'robot_1_global_y', 'robot_1_global_z']].values
        robot1_radar = data[['robot_1_global_x_radar', 'robot_1_global_y_radar', 'robot_1_global_z_radar']].values
        
        robot2_position = data[['robot_2_global_x', 'robot_2_global_y', 'robot_2_global_z']].values
        robot2_radar = data[['robot_2_global_x_radar', 'robot_2_global_y_radar', 'robot_2_global_z_radar']].values
        
        # Calculate RMSE separately for each robot
        rmse_robot1 = np.sqrt(mean_squared_error(robot1_position, robot1_radar))
        rmse_robot2 = np.sqrt(mean_squared_error(robot2_position, robot2_radar))
        
        # Calculate RMSE for specific axes
        rmse_robot1_x = np.sqrt(mean_squared_error(robot1_position[:, 0], robot1_radar[:, 0]))
        rmse_robot1_y = np.sqrt(mean_squared_error(robot1_position[:, 1], robot1_radar[:, 1]))
        rmse_robot1_z = np.sqrt(mean_squared_error(robot1_position[:, 2], robot1_radar[:, 2]))
        
        rmse_robot2_x = np.sqrt(mean_squared_error(robot2_position[:, 0], robot2_radar[:, 0]))
        rmse_robot2_y = np.sqrt(mean_squared_error(robot2_position[:, 1], robot2_radar[:, 1]))
        rmse_robot2_z = np.sqrt(mean_squared_error(robot2_position[:, 2], robot2_radar[:, 2]))
        
        # Store metrics
        metrics = {
            'rmse_robot1': rmse_robot1,
            'rmse_robot2': rmse_robot2,
            'rmse_robot1_x': rmse_robot1_x,
            'rmse_robot1_y': rmse_robot1_y,
            'rmse_robot1_z': rmse_robot1_z,
            'rmse_robot2_x': rmse_robot2_x,
            'rmse_robot2_y': rmse_robot2_y,
            'rmse_robot2_z': rmse_robot2_z,
            'sample_size': len(data)
        }
        
        print(f"RMSE Evaluation (n={len(data)} points)")
        print(f"  Robot 1 RMSE: {rmse_robot1:.4f} m")
        print(f"    X-axis: {rmse_robot1_x:.4f} m")
        print(f"    Y-axis: {rmse_robot1_y:.4f} m")
        print(f"    Z-axis: {rmse_robot1_z:.4f} m")
        print(f"  Robot 2 RMSE: {rmse_robot2:.4f} m")
        print(f"    X-axis: {rmse_robot2_x:.4f} m")
        print(f"    Y-axis: {rmse_robot2_y:.4f} m")
        print(f"    Z-axis: {rmse_robot2_z:.4f} m")
        
        self.evaluation_results['rmse'] = metrics
        return metrics
    
    def calculate_point_cloud_metrics(self, threshold=0.5):
        """
        Calculate precision, recall, and F1-score for point cloud registration.
        
        Args:
            threshold (float): Distance threshold in meters
            
        Returns:
            dict: Point cloud metrics
        """
        if self.data is None:
            print("Error: No data loaded")
            return None
            
        # Create point clouds from both robots
        robot1_points = self.data.dropna(subset=[
            'robot_1_global_x_radar', 'robot_1_global_y_radar', 'robot_1_global_z_radar'
        ])[['robot_1_global_x_radar', 'robot_1_global_y_radar', 'robot_1_global_z_radar']].values
        
        robot2_points = self.data.dropna(subset=[
            'robot_2_global_x_radar', 'robot_2_global_y_radar', 'robot_2_global_z_radar'
        ])[['robot_2_global_x_radar', 'robot_2_global_y_radar', 'robot_2_global_z_radar']].values
        
        if len(robot1_points) == 0 or len(robot2_points) == 0:
            print("Error: Not enough points for evaluation")
            return None
            
        # Build KD-trees for efficient nearest neighbor search
        tree1 = cKDTree(robot1_points)
        tree2 = cKDTree(robot2_points)
        
        # For each point in robot1, find the nearest point in robot2
        dist_1to2, _ = tree1.query(robot2_points, k=1)
        # For each point in robot2, find the nearest point in robot1
        dist_2to1, _ = tree2.query(robot1_points, k=1)
        
        # Calculate metrics
        true_positive_1to2 = np.sum(dist_1to2 < threshold)
        precision = true_positive_1to2 / len(robot2_points) if len(robot2_points) > 0 else 0
        
        true_positive_2to1 = np.sum(dist_2to1 < threshold)
        recall = true_positive_2to1 / len(robot1_points) if len(robot1_points) > 0 else 0
        
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate distances statistics
        distance_stats = {
            'mean_distance': np.mean(dist_1to2),
            'median_distance': np.median(dist_1to2),
            'std_distance': np.std(dist_1to2),
            'min_distance': np.min(dist_1to2),
            'max_distance': np.max(dist_1to2),
            'percentile_25': np.percentile(dist_1to2, 25),
            'percentile_75': np.percentile(dist_1to2, 75),
        }
        
        # Store metrics
        metrics = {
            'threshold': threshold,
            'robot1_points': len(robot1_points),
            'robot2_points': len(robot2_points),
            'true_positive_1to2': true_positive_1to2,
            'true_positive_2to1': true_positive_2to1,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'distance_stats': distance_stats
        }
        
        print(f"Point Cloud Metrics (threshold: {threshold}m)")
        print(f"  Robot 1 points: {len(robot1_points)}")
        print(f"  Robot 2 points: {len(robot2_points)}")
        print(f"  True positives (1→2): {true_positive_1to2}")
        print(f"  True positives (2→1): {true_positive_2to1}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-score: {f1_score:.4f}")
        print(f"  Distance statistics:")
        for key, value in distance_stats.items():
            print(f"    {key}: {value:.4f} m")
        
        self.evaluation_results['point_cloud_metrics'] = metrics
        return metrics
    
    def plot_error_distribution(self):
        """
        Plot the distribution of errors in the transformed point clouds.
        
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        if self.data is None:
            print("Error: No data loaded")
            return None
            
        # Filter out rows with NaN values
        data = self.data.dropna(subset=[
            'robot_1_global_x', 'robot_1_global_y', 'robot_1_global_z',
            'robot_1_global_x_radar', 'robot_1_global_y_radar', 'robot_1_global_z_radar',
            'robot_2_global_x', 'robot_2_global_y', 'robot_2_global_z',
            'robot_2_global_x_radar', 'robot_2_global_y_radar', 'robot_2_global_z_radar'
        ])
        
        if len(data) == 0:
            print("Error: No valid data for error distribution")
            return None
            
        # Calculate errors for each robot
        robot1_error_x = data['robot_1_global_x'] - data['robot_1_global_x_radar']
        robot1_error_y = data['robot_1_global_y'] - data['robot_1_global_y_radar']
        robot1_error_z = data['robot_1_global_z'] - data['robot_1_global_z_radar']
        
        robot2_error_x = data['robot_2_global_x'] - data['robot_2_global_x_radar']
        robot2_error_y = data['robot_2_global_y'] - data['robot_2_global_y_radar']
        robot2_error_z = data['robot_2_global_z'] - data['robot_2_global_z_radar']
        
        # Calculate Euclidean error
        robot1_error = np.sqrt(robot1_error_x**2 + robot1_error_y**2 + robot1_error_z**2)
        robot2_error = np.sqrt(robot2_error_x**2 + robot2_error_y**2 + robot2_error_z**2)
        
        # Create figure
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # Error histograms for x, y, z components
        ax = axs[0, 0]
        ax.hist(robot1_error_x, bins=30, alpha=0.5, label='Robot 1 X-error')
        ax.hist(robot2_error_x, bins=30, alpha=0.5, label='Robot 2 X-error')
        ax.set_xlabel('X Error (m)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        ax = axs[0, 1]
        ax.hist(robot1_error_y, bins=30, alpha=0.5, label='Robot 1 Y-error')
        ax.hist(robot2_error_y, bins=30, alpha=0.5, label='Robot 2 Y-error')
        ax.set_xlabel('Y Error (m)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Euclidean error histogram
        ax = axs[1, 0]
        ax.hist(robot1_error, bins=30, alpha=0.5, label='Robot 1')
        ax.hist(robot2_error, bins=30, alpha=0.5, label='Robot 2')
        ax.set_xlabel('Euclidean Error (m)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Error scatter plot (X vs Y error)
        ax = axs[1, 1]
        ax.scatter(robot1_error_x, robot1_error_y, alpha=0.5, label='Robot 1')
        ax.scatter(robot2_error_x, robot2_error_y, alpha=0.5, label='Robot 2')
        ax.set_xlabel('X Error (m)')
        ax.set_ylabel('Y Error (m)')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add reference lines
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle('Error Distribution Analysis', fontsize=16)
        plt.subplots_adjust(top=0.95)
        
        return fig
    
    def plot_error_heatmap(self, resolution=0.5):
        """
        Create a heatmap showing the spatial distribution of errors.
        
        Args:
            resolution (float): Grid cell size in meters
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        if self.data is None:
            print("Error: No data loaded")
            return None
            
        # Filter out rows with NaN values
        data = self.data.dropna(subset=[
            'robot_1_global_x', 'robot_1_global_y',
            'robot_1_global_x_radar', 'robot_1_global_y_radar',
            'robot_2_global_x', 'robot_2_global_y',
            'robot_2_global_x_radar', 'robot_2_global_y_radar'
        ])
        
        if len(data) == 0:
            print("Error: No valid data for error heatmap")
            return None
            
        # Define grid boundaries
        x_min, x_max = -10.0, 11.0
        y_min, y_max = -5.0, 6.0
        
        # Create grids
        x_bins = np.arange(x_min, x_max + resolution, resolution)
        y_bins = np.arange(y_min, y_max + resolution, resolution)
        
        # Calculate errors
        data['robot1_error'] = np.sqrt(
            (data['robot_1_global_x'] - data['robot_1_global_x_radar'])**2 +
            (data['robot_1_global_y'] - data['robot_1_global_y_radar'])**2
        )
        
        data['robot2_error'] = np.sqrt(
            (data['robot_2_global_x'] - data['robot_2_global_x_radar'])**2 +
            (data['robot_2_global_y'] - data['robot_2_global_y_radar'])**2
        )
        
        # Create figure
        fig, axs = plt.subplots(1, 2, figsize=(16, 7))
        
        # Robot 1 error heatmap
        ax = axs[0]
        heatmap_robot1, _, _, im1 = ax.hist2d(
            data['robot_1_global_x_radar'], 
            data['robot_1_global_y_radar'],
            bins=[x_bins, y_bins],
            weights=data['robot1_error'],
            cmap='hot',
            norm=plt.cm.colors.LogNorm()
        )
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Robot 1 Error Heatmap')
        plt.colorbar(im1, ax=ax, label='Error (m)')
        
        # Robot 2 error heatmap
        ax = axs[1]
        heatmap_robot2, _, _, im2 = ax.hist2d(
            data['robot_2_global_x_radar'], 
            data['robot_2_global_y_radar'],
            bins=[x_bins, y_bins],
            weights=data['robot2_error'],
            cmap='hot',
            norm=plt.cm.colors.LogNorm()
        )
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Robot 2 Error Heatmap')
        plt.colorbar(im2, ax=ax, label='Error (m)')
        
        plt.tight_layout()
        
        return fig
    
    def run_full_evaluation(self):
        """
        Run all evaluation metrics and compile a comprehensive report.
        
        Returns:
            dict: Complete evaluation results
        """
        print("Running complete transformation evaluation...")
        
        # Run all evaluation methods
        self.calculate_rmse()
        self.calculate_iou_2d()
        self.calculate_point_cloud_metrics()
        
        # Compile results
        all_results = {
            'dataset': os.path.basename(self.data_path),
            'data_points': len(self.data),
            'rmse': self.evaluation_results.get('rmse', {}),
            'iou_2d': self.evaluation_results.get('iou_2d', {}),
            'point_cloud_metrics': self.evaluation_results.get('point_cloud_metrics', {})
        }
        
        # Print summary
        print("\nEvaluation Summary:")
        print(f"Dataset: {all_results['dataset']}")
        print(f"Total data points: {all_results['data_points']}")
        
        if 'rmse' in self.evaluation_results:
            print(f"Overall RMSE: Robot 1 = {self.evaluation_results['rmse']['rmse_robot1']:.4f}m, " +
                 f"Robot 2 = {self.evaluation_results['rmse']['rmse_robot2']:.4f}m")
            
        if 'iou_2d' in self.evaluation_results:
            print(f"2D IoU: {self.evaluation_results['iou_2d']['iou']:.4f}")
            
        if 'point_cloud_metrics' in self.evaluation_results:
            print(f"Point Cloud F1-Score: {self.evaluation_results['point_cloud_metrics']['f1_score']:.4f}")
        
        return all_results
    
    def save_results(self, output_path=None):
        """
        Save evaluation results to a JSON file.
        
        Args:
            output_path (str, optional): Path to save the results
            
        Returns:
            str: Path where results were saved
        """
        import json
        
        if not self.evaluation_results:
            print("Error: No evaluation results to save")
            return None
            
        if output_path is None:
            # Create output path based on input path
            if self.data_path:
                path_obj = Path(self.data_path)
                output_path = str(path_obj.parent / f"evaluation_{path_obj.stem}.json")
            else:
                output_path = "evaluation_results.json"
                
        # Convert numpy values to Python types for JSON serialization
        results_json = {}
        
        for key, value in self.evaluation_results.items():
            if isinstance(value, dict):
                results_json[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        results_json[key][k] = v.tolist()
                    elif isinstance(v, np.number):
                        results_json[key][k] = float(v)
                    else:
                        results_json[key][k] = v
            else:
                if isinstance(value, np.ndarray):
                    results_json[key] = value.tolist()
                elif isinstance(value, np.number):
                    results_json[key] = float(value)
                else:
                    results_json[key] = value
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(results_json, f, indent=2)
            
        print(f"Saved evaluation results to: {output_path}")
        return output_path


def main():
    """Main function to run the evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate point cloud transformation accuracy')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to transformed data CSV file')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Path to save evaluation results (default: auto-generated)')
    parser.add_argument('--plots', '-p', action='store_true',
                        help='Generate and display error plots')
    parser.add_argument('--save-plots', '-s', type=str, default=None,
                        help='Path to save plot images (default: not saved)')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = TransformationEvaluator()
    
    # Load data
    if not evaluator.load_data(args.input):
        return 1
    
    # Run evaluation
    evaluator.run_full_evaluation()
    
    # Save results
    evaluator.save_results(args.output)
    
    # Generate plots if requested
    if args.plots or args.save_plots:
        # Error distribution plot
        err_dist_fig = evaluator.plot_error_distribution()
        
        # Error heatmap
        err_heatmap_fig = evaluator.plot_error_heatmap()
        
        # Save plots if requested
        if args.save_plots:
            base_path = Path(args.save_plots)
            base_path.mkdir(parents=True, exist_ok=True)
            
            err_dist_path = base_path / "error_distribution.png"
            err_heatmap_path = base_path / "error_heatmap.png"
            
            if err_dist_fig:
                err_dist_fig.savefig(err_dist_path, dpi=300, bbox_inches='tight')
                print(f"Saved error distribution plot to: {err_dist_path}")
                
            if err_heatmap_fig:
                err_heatmap_fig.savefig(err_heatmap_path, dpi=300, bbox_inches='tight')
                print(f"Saved error heatmap to: {err_heatmap_path}")
        
        # Display plots if requested
        if args.plots:
            plt.show()
    
    return 0


if __name__ == "__main__":
    main()
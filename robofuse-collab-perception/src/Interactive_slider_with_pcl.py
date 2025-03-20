#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import argparse
import os
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
import json
import glob
import sys

class RoboFUSEVisualizer:
    """
    Interactive visualization tool for RoboFUSE dataset that shows robot movements
    and point clouds with a time slider.
    """
    
    def __init__(self, data_path=None):
        """
        Initialize the visualizer.
        
        Args:
            data_path (str): Path to the transformed data CSV file
        """
        self.data_path = data_path
        self.data = None
        self.current_frame = 0
        self.playing = False
        self.animation_speed = 1.0
        self.show_pcl = True
        self.show_trails = True
        self.plot_type = '2d'
        
        # Warehouse boundaries
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
        
        # Colors for visualization
        self.colors = {
            'robot_1': 'blue',
            'robot_2': 'red',
            'robot_1_pcl': 'lightblue',
            'robot_2_pcl': 'salmon',
            'workstation': 'gray',
            'boundary': 'black'
        }
        
        # Trail data for robots
        self.trail_length = 50
        self.robot_trails = {'robot_1': [], 'robot_2': []}
        
        # Load evaluation metrics if available
        self.evaluation_results = None
        
        # For Vicon data mode
        self.frames = []  # List of frames with timestamp and data
        self.robot_names = ['ep03', 'ep05']  # Focus on these robots
        self.workstation_names = ['AS_1_neu', 'AS_3_neu', 'AS_4_neu', 'AS_5_neu', 'AS_6_neu']
        self.frame_count = 0
        self.robot_markers = {}
        self.workstation_positions = {}  # Positions by timestamp
        self.workstation_patches = {}
        self.workstation_colors = {
            'AS_1_neu': 'lightcoral',
            'AS_3_neu': 'lightblue',
            'AS_4_neu': 'lightgreen',
            'AS_5_neu': 'lightyellow',
            'AS_6_neu': 'lightpink'
        }
        
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
        
        # Check if this is a raw Vicon file
        if self.data_path.endswith('.txt'):
            return self.parse_vicon_file(self.data_path)
        
        try:
            print(f"Loading transformed data from: {self.data_path}")
            self.data = pd.read_csv(self.data_path)
            
            # Try to load evaluation results if they exist
            eval_path = os.path.join(os.path.dirname(self.data_path), "evaluation_results.json")
            if os.path.exists(eval_path):
                try:
                    with open(eval_path, 'r') as f:
                        self.evaluation_results = json.load(f)
                    print(f"Loaded evaluation results from {eval_path}")
                except Exception as e:
                    print(f"Could not load evaluation results: {e}")
            
            # Check for required columns and adapt to naming variations
            # Standard column names we're looking for
            std_cols = {
                'timestamp': ['vicon_timestamp', 'timestamp', 'time'],
                'robot1_pos_x': ['ep03_pos_x', 'robot_1_pos_x', 'robot_1_global_x'],
                'robot1_pos_y': ['ep03_pos_y', 'robot_1_pos_y', 'robot_1_global_y'],
                'robot1_pos_z': ['ep03_pos_z', 'robot_1_pos_z', 'robot_1_global_z'],
                'robot2_pos_x': ['ep05_pos_x', 'robot_2_pos_x', 'robot_2_global_x'],
                'robot2_pos_y': ['ep05_pos_y', 'robot_2_pos_y', 'robot_2_global_y'],
                'robot2_pos_z': ['ep05_pos_z', 'robot_2_pos_z', 'robot_2_global_z'],
                'robot1_pcl_x': ['robot_1_global_x_radar', 'robot_1_x_global'],
                'robot1_pcl_y': ['robot_1_global_y_radar', 'robot_1_y_global'],
                'robot1_pcl_z': ['robot_1_global_z_radar', 'robot_1_z_global'],
                'robot2_pcl_x': ['robot_2_global_x_radar', 'robot_2_x_global'],
                'robot2_pcl_y': ['robot_2_global_y_radar', 'robot_2_y_global'],
                'robot2_pcl_z': ['robot_2_global_z_radar', 'robot_2_z_global']
            }
            
            # Create a mapping from actual column names to standard names
            col_mapping = {}
            for std_col, variations in std_cols.items():
                if std_col in self.data.columns:
                    # Already has standard name
                    continue
                
                # Check if any variation exists
                for var in variations:
                    if var in self.data.columns:
                        col_mapping[var] = std_col
                        break
            
            # Rename columns if needed
            if col_mapping:
                self.data = self.data.rename(columns=col_mapping)
                print(f"Renamed {len(col_mapping)} columns to standard names")
            
            # Create standardized column names if not already present
            for std_col, variations in std_cols.items():
                if std_col not in self.data.columns:
                    # Find the first variation that exists
                    for var in variations:
                        if var in self.data.columns:
                            self.data[std_col] = self.data[var]
                            break
            
            # Ensure we have a timestamp column
            if 'timestamp' not in self.data.columns:
                # If we don't have a timestamp, create an artificial one
                print("Warning: No timestamp column found, creating artificial timestamps")
                self.data['timestamp'] = np.arange(len(self.data))
            
            # Extract unique timestamps for slider
            self.timestamps = self.data['timestamp'].unique()
            self.timestamps.sort()
            
            # Filter data to remove NaN values in critical columns
            req_cols = ['robot1_pos_x', 'robot1_pos_y', 'robot2_pos_x', 'robot2_pos_y']
            
            # Check if PCL columns should be included in required columns
            pcl_req = []
            for col in ['robot1_pcl_x', 'robot1_pcl_y', 'robot2_pcl_x', 'robot2_pcl_y']:
                if col in self.data.columns:
                    pcl_req.append(col)
            
            self.data = self.data.dropna(subset=req_cols)
            print(f"Loaded {len(self.data)} valid data points across {len(self.timestamps)} timestamps")
            
            # Pre-compute robot trails for visualization
            self.calculate_robot_trails()
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
            
    def calculate_robot_trails(self):
        """Pre-calculate position trails for transformed data visualization"""
        # Reset trail data
        self.robot_trails = {'robot_1': [], 'robot_2': []}
        
        # Initialize with positions from the first frame
        if not self.data.empty and len(self.timestamps) > 0:
            first_frame = self.data[self.data['timestamp'] == self.timestamps[0]]
            if not first_frame.empty:
                # Add initial positions
                if 'robot1_pos_x' in first_frame.columns and 'robot1_pos_y' in first_frame.columns:
                    r1_pos_x = first_frame['robot1_pos_x'].iloc[0]
                    r1_pos_y = first_frame['robot1_pos_y'].iloc[0]
                    self.robot_trails['robot_1'].append((r1_pos_x, r1_pos_y))
                    
                if 'robot2_pos_x' in first_frame.columns and 'robot2_pos_y' in first_frame.columns:
                    r2_pos_x = first_frame['robot2_pos_x'].iloc[0]
                    r2_pos_y = first_frame['robot2_pos_y'].iloc[0]
                    self.robot_trails['robot_2'].append((r2_pos_x, r2_pos_y))
            
    def parse_vicon_file(self, file_path):
        """Parse Vicon data file to extract robot and workstation positions"""
        print(f"Processing Vicon file: {os.path.basename(file_path)}...")
        
        # Data storage by timestamp
        frame_data = {}
        workstation_data = {}
        
        try:
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f):
                    try:
                        # Parse line content
                        data = eval(line.strip())
                        
                        # Skip empty data
                        if all(all(v == 0 for v in obj_data.values()) for obj_data in data.values()):
                            continue
                        
                        # Process objects in this frame
                        timestamp = None
                        robots = {}
                        workstations = {}
                        
                        for key, values in data.items():
                            # Skip empty entries
                            if all(v == 0 for v in values.values()):
                                continue
                                
                            # Extract object name
                            object_name = key.split('/')[1]
                            
                            # Get timestamp (convert to seconds)
                            if timestamp is None:
                                timestamp = values['system_time'] / 1e9
                            
                            # Store position data
                            position_data = {
                                'x': values['pos_x'],
                                'y': values['pos_y'],
                                'z': values['pos_z'],
                                'yaw': values['yaw'],
                                'rot_x': values['rot_x'],
                                'rot_y': values['rot_y'],
                                'rot_z': values['rot_z']
                            }
                            
                            # Sort into robot vs workstation
                            if object_name.startswith('AS_'):
                                workstations[object_name] = position_data
                            elif object_name in self.robot_names:
                                robots[object_name] = position_data
                        
                        # Store frame if it has valid data
                        if timestamp and (robots or workstations):
                            # Add to frames
                            if robots:
                                if timestamp not in frame_data:
                                    frame_data[timestamp] = {}
                                frame_data[timestamp].update(robots)
                            
                            # Add to workstation data
                            if workstations:
                                if timestamp not in workstation_data:
                                    workstation_data[timestamp] = {}
                                workstation_data[timestamp].update(workstations)
                            
                    except Exception as e:
                        print(f"Error parsing line {line_num}: {e}")
                        continue
        except Exception as e:
            print(f"Error reading file: {e}")
            return False
        
        # Check if we found any valid data
        if not frame_data:
            print("No valid robot data found!")
            return False
            
        # Sort frames by timestamp
        sorted_times = sorted(frame_data.keys())
        self.frames = [{'timestamp': t, 'robots': frame_data[t]} for t in sorted_times]
        self.frame_count = len(self.frames)
        
        # Pre-process workstation positions
        self.process_workstation_positions(workstation_data)
        
        # Calculate robot trails
        self.build_robot_trails()
        
        # Extract timestamps for slider
        self.timestamps = sorted_times
        
        print(f"Processed {self.frame_count} frames")
        print(f"Found robots: {', '.join(set().union(*[frame['robots'].keys() for frame in self.frames]))}")
        print(f"Found workstations: {', '.join(self.workstation_positions.keys())}")
        
        return True
    
    def process_workstation_positions(self, workstation_data):
        """Process and organize workstation position data"""
        # First check for dedicated workstation files
        ws_positions_dir = "/media/yugi/MAIN DRIVE/RoboFUSE_Dataset/CPPS_Static_Scenario/archiv_measurements/Vicon/working_station_positions"
        
        if os.path.exists(ws_positions_dir):
            print(f"Searching for workstation positions in {ws_positions_dir}")
            ws_files = glob.glob(os.path.join(ws_positions_dir, "*.txt"))
            
            if ws_files:
                # Sort files by timestamp (as embedded in filenames)
                ws_files.sort(key=lambda x: os.path.getmtime(x))
                
                # Find the file with the closest timestamp to our vicon data
                if self.frames:
                    vicon_start_time = self.frames[0]['timestamp']
                    closest_file = None
                    min_time_diff = float('inf')
                    
                    for ws_file in ws_files:
                        file_time = os.path.getmtime(ws_file)
                        time_diff = abs(file_time - vicon_start_time)
                        
                        if time_diff < min_time_diff:
                            min_time_diff = time_diff
                            closest_file = ws_file
                    
                    if closest_file:
                        print(f"Using workstation positions from {os.path.basename(closest_file)}")
                        workstation_positions = self.parse_workstation_file(closest_file)
                        
                        if workstation_positions:
                            self.workstation_positions = workstation_positions
                            return
                        
            print("No suitable workstation files found, checking vicon data...")
        
        # If no workstation files found or parsing failed, fall back to vicon data
        # If no workstation data
        if not workstation_data:
            print("No workstation data found, using defaults")
            # Default positions as fallback
            self.workstation_positions = {
                'AS_1_neu': {'x': 1.52, 'y': 2.24, 'z': 1.02, 'yaw': 0},
                'AS_3_neu': {'x': -5.74, 'y': -0.13, 'z': 1.47, 'yaw': 0},
                'AS_4_neu': {'x': 5.37, 'y': 0.21, 'z': 2.30, 'yaw': np.pi/2},  # 90 degree rotation
                'AS_5_neu': {'x': -3.05, 'y': 2.39, 'z': 2.21, 'yaw': 0},
                'AS_6_neu': {'x': 0.01, 'y': -1.45, 'z': 1.53, 'yaw': 0}
            }
            return
        
        # Calculate average positions for each workstation
        avg_positions = {}
        
        for ws_name in self.workstation_names:
            positions = []
            
            # Collect all positions for this workstation
            for timestamp, ws_data in workstation_data.items():
                if ws_name in ws_data:
                    positions.append(ws_data[ws_name])
            
            if positions:
                # Calculate average position
                avg_pos = {
                    'x': np.mean([p['x'] for p in positions]),
                    'y': np.mean([p['y'] for p in positions]),
                    'z': np.mean([p['z'] for p in positions]),
                    'yaw': np.mean([p['yaw'] for p in positions])
                }
                
                # Apply 90-degree rotation to AS_4_neu
                if ws_name == 'AS_4_neu':
                    avg_pos['yaw'] = np.pi/2  # 90 degrees in radians
                
                avg_positions[ws_name] = avg_pos
                print(f"Average position for {ws_name}: ({avg_pos['x']:.2f}, {avg_pos['y']:.2f})")
        
        self.workstation_positions = avg_positions

    def parse_workstation_file(self, file_path):
        """Parse a workstation positions file"""
        try:
            workstation_positions = {}
            
            with open(file_path, 'r') as f:
                for line in f:
                    try:
                        # Parse line content
                        data = eval(line.strip())
                        
                        # Process each object in the data
                        for key, values in data.items():
                            # Extract object name 
                            object_name = key.split('/')[1]
                            
                            # Only process workstations
                            if object_name.startswith('AS_'):
                                # Store position data
                                workstation_positions[object_name] = {
                                    'x': values['pos_x'],
                                    'y': values['pos_y'],
                                    'z': values['pos_z'],
                                    'yaw': values['yaw'],
                                    'rot_x': values['rot_x'],
                                    'rot_y': values['rot_y'],
                                    'rot_z': values['rot_z']
                                }
                                
                                # Apply 90-degree rotation to AS_4_neu
                                if object_name == 'AS_4_neu':
                                    workstation_positions[object_name]['yaw'] = np.pi/2
                                
                                print(f"Found workstation {object_name} at ({values['pos_x']:.2f}, {values['pos_y']:.2f})")
                        
                    except Exception as e:
                        print(f"Error parsing line in workstation file: {e}")
                        continue
            
            return workstation_positions
        
        except Exception as e:
            print(f"Error reading workstation file: {e}")
            return None
   
    def build_robot_trails(self):
        """Pre-calculate position trails for robots in Vicon data"""
        print("Building robot trails...")
        
        # For each robot, collect positions over time
        robot_positions = {robot: [] for robot in self.robot_names}
        
        # Extract position data
        for frame in self.frames:
            for robot, data in frame['robots'].items():
                robot_positions[robot].append((data['x'], data['y']))
        
        # Pre-compute trails for each frame
        for i in range(self.frame_count):
            frame = self.frames[i]
            frame['trails'] = {}
            
            for robot in self.robot_names:
                positions = robot_positions[robot][:i+1]
                
                if positions:
                    # Use last N positions for trail
                    trail = positions[-self.trail_length:] if len(positions) > self.trail_length else positions
                    frame['trails'][robot] = trail
    
    def setup_visualization(self):
        """
        Set up the visualization figure and interactive controls.
        
        Returns:
            tuple: Figure and axes objects
        """
        # Create figure and main plot area
        self.fig = plt.figure(figsize=(14, 10))
        
        # Adjust layout to make room for controls
        self.fig.subplots_adjust(left=0.1, bottom=0.25, right=0.95, top=0.95)
        
        # Create the axes based on plot type
        if self.plot_type == '3d':
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.set_zlim([self.boundaries['z_min'], self.boundaries['z_max']])
            self.ax.set_zlabel('Z (m)')
        else:
            self.ax = self.fig.add_subplot(111)
            self.ax.set_aspect('equal')
        
        # Common settings
        self.ax.set_xlim([self.boundaries['x_min'], self.boundaries['x_max']])
        self.ax.set_ylim([self.boundaries['y_min'], self.boundaries['y_max']])
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        
        # Draw warehouse boundaries in 2D mode
        if self.plot_type == '2d':
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
            self.ax.plot(boundary_x, boundary_y, 'k-', linewidth=2, label='Warehouse Boundary')
            
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
                self.ax.add_patch(rect)
                self.ax.text(pos['x'], pos['y'], name, 
                      horizontalalignment='center', verticalalignment='center',
                      size=8, zorder=3)
        
        # Create plot elements for robots
        if self.plot_type == '3d':
            # Robot position markers
            self.robot1_pos = self.ax.scatter([], [], [], color=self.colors['robot_1'], 
                                        marker='^', s=100, label='Robot 1')
            self.robot2_pos = self.ax.scatter([], [], [], color=self.colors['robot_2'], 
                                        marker='^', s=100, label='Robot 2')
            
            # Point cloud markers
            self.robot1_pcl = self.ax.scatter([], [], [], color=self.colors['robot_1_pcl'], 
                                        alpha=0.5, s=20, label='Robot 1 PCL')
            self.robot2_pcl = self.ax.scatter([], [], [], color=self.colors['robot_2_pcl'], 
                                        alpha=0.5, s=20, label='Robot 2 PCL')
            
            # Direction arrows (not well supported in 3D)
            self.robot1_arrow = None
            self.robot2_arrow = None
            
            # Trails not well supported in 3D
            self.robot1_trail = None
            self.robot2_trail = None
            
        else:
            # Robot position markers
            self.robot1_pos = self.ax.scatter([], [], color=self.colors['robot_1'], 
                                        marker='^', s=100, label='Robot 1')
            self.robot2_pos = self.ax.scatter([], [], color=self.colors['robot_2'], 
                                        marker='^', s=100, label='Robot 2')
            
            # Direction arrows
            self.robot1_arrow = self.ax.quiver([], [], [], [], color=self.colors['robot_1'], 
                                    scale=5, width=0.008, zorder=10)
            self.robot2_arrow = self.ax.quiver([], [], [], [], color=self.colors['robot_2'], 
                                    scale=5, width=0.008, zorder=10)
            
            # Point cloud markers
            self.robot1_pcl = self.ax.scatter([], [], color=self.colors['robot_1_pcl'], 
                                        alpha=0.5, s=20, label='Robot 1 PCL')
            self.robot2_pcl = self.ax.scatter([], [], color=self.colors['robot_2_pcl'], 
                                        alpha=0.5, s=20, label='Robot 2 PCL')
            
            # Robot trails
            self.robot1_trail, = self.ax.plot([], [], color=self.colors['robot_1'], 
                                        alpha=0.5, linewidth=1.5)
            self.robot2_trail, = self.ax.plot([], [], color=self.colors['robot_2'], 
                                        alpha=0.5, linewidth=1.5)
        
        # Position text labels
        self.robot1_text = self.ax.text(0, 0, '', fontsize=9, 
                          bbox=dict(facecolor='white', alpha=0.7))
        self.robot2_text = self.ax.text(0, 0, '', fontsize=9, 
                          bbox=dict(facecolor='white', alpha=0.7))
            
        # Add time and metrics display
        self.time_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes, 
                                fontsize=10, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Add metrics display if available
        if self.evaluation_results:
            metrics_text = "Metrics:\n"
            if 'rmse' in self.evaluation_results:
                rmse = self.evaluation_results['rmse']
                metrics_text += f"RMSE R1: {rmse.get('rmse_robot1', 0):.3f}m\n"
                metrics_text += f"RMSE R2: {rmse.get('rmse_robot2', 0):.3f}m\n"
            
            if 'iou_2d' in self.evaluation_results:
                iou = self.evaluation_results['iou_2d']
                metrics_text += f"IoU: {iou.get('iou', 0):.3f}\n"
                
            self.metrics_text = self.ax.text(0.82, 0.98, metrics_text, transform=self.ax.transAxes,
                                      fontsize=9, verticalalignment='top',
                                      bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        
        # Add title
        if self.data is not None:
            scenario = os.path.basename(os.path.dirname(self.data_path))
            dataset = os.path.basename(os.path.dirname(os.path.dirname(self.data_path)))
            if scenario and dataset:
                plt.title(f'RoboFUSE Visualization: {dataset} - {scenario}')
            else:
                plt.title(f'RoboFUSE Visualization: {os.path.basename(self.data_path)}')
        elif self.frames:
            plt.title(f'Vicon Visualization: {os.path.basename(self.data_path)}')
        else:
            plt.title('RoboFUSE Visualization')
            
        self.ax.legend(loc='upper right')
        self.ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add time slider
        slider_ax = self.fig.add_axes([0.25, 0.1, 0.65, 0.03])
        max_frame = max(1, len(self.timestamps)-1) if hasattr(self, 'timestamps') and self.timestamps is not None else max(1, self.frame_count-1)
        self.slider = Slider(
            slider_ax, 'Time', 0, max_frame,
            valinit=0, valstep=1
        )
        self.slider.on_changed(self.update_time)
        
        # Add play/pause button
        play_ax = self.fig.add_axes([0.08, 0.1, 0.1, 0.03])
        self.play_button = Button(play_ax, 'Play', color='lightgoldenrodyellow')
        self.play_button.on_clicked(self.toggle_play)
        
        # Add speed control buttons
        speed_slower_ax = self.fig.add_axes([0.05, 0.15, 0.07, 0.03])
        speed_faster_ax = self.fig.add_axes([0.13, 0.15, 0.07, 0.03])
        self.speed_slower = Button(speed_slower_ax, 'Slower', color='lightblue')
        self.speed_faster = Button(speed_faster_ax, 'Faster', color='lightpink')
        self.speed_slower.on_clicked(self.decrease_speed)
        self.speed_faster.on_clicked(self.increase_speed)
        
        # Add view toggle (2D/3D)
        view_ax = self.fig.add_axes([0.05, 0.05, 0.15, 0.03])
        self.view_button = Button(view_ax, f'View: {self.plot_type.upper()}', color='lightgreen')
        self.view_button.on_clicked(self.toggle_view)
        
        # Add PCL visibility toggle
        pcl_ax = self.fig.add_axes([0.25, 0.05, 0.15, 0.03])
        self.pcl_button = Button(pcl_ax, 'PCL: ON' if self.show_pcl else 'PCL: OFF', color='lightgreen')
        self.pcl_button.on_clicked(self.toggle_pcl)
        
        # Add trail visibility toggle
        trail_ax = self.fig.add_axes([0.45, 0.05, 0.15, 0.03])
        self.trail_button = Button(trail_ax, 'Trails: ON' if self.show_trails else 'Trails: OFF', color='lightgreen')
        self.trail_button.on_clicked(self.toggle_trails)
        
        return self.fig, self.ax
    
    def update_visualization(self, frame_idx=None):
        """
        Update the visualization for a specific time frame.
        
        Args:
            frame_idx (int): Index into self.timestamps for the frame to display
        """
        if frame_idx is not None:
            self.current_frame = frame_idx
        
        # Check if we're using transformed data or Vicon data
        if self.data is not None:
            self.update_transformed_visualization()
        elif self.frames:
            self.update_vicon_visualization()
    
    def update_transformed_visualization(self):
        """Update visualization for transformed data"""
        # Get current timestamp
        timestamp = self.timestamps[self.current_frame]
        
        # Get data for this timestamp
        current_data = self.data[self.data['timestamp'] == timestamp]
        
        if current_data.empty:
            print(f"Warning: No data for timestamp {timestamp}")
            return
        
        # Calculate relative time for display
        rel_time = timestamp - self.timestamps[0]
        self.time_text.set_text(f'Time: {rel_time:.2f}s')
        
        # Get positions for this timestep (use first row if multiple rows have same timestamp)
        r1_pos_x = current_data['robot1_pos_x'].iloc[0]
        r1_pos_y = current_data['robot1_pos_y'].iloc[0]
        r1_pos_z = current_data['robot1_pos_z'].iloc[0] if 'robot1_pos_z' in current_data.columns else 0
        
        r2_pos_x = current_data['robot2_pos_x'].iloc[0]
        r2_pos_y = current_data['robot2_pos_y'].iloc[0]
        r2_pos_z = current_data['robot2_pos_z'].iloc[0] if 'robot2_pos_z' in current_data.columns else 0
        
        # Update robot positions
        if self.plot_type == '3d':
            self.robot1_pos._offsets3d = ([r1_pos_x], [r1_pos_y], [r1_pos_z])
            self.robot2_pos._offsets3d = ([r2_pos_x], [r2_pos_y], [r2_pos_z])
        else:
            self.robot1_pos.set_offsets(np.column_stack([r1_pos_x, r1_pos_y]))
            self.robot2_pos.set_offsets(np.column_stack([r2_pos_x, r2_pos_y]))

        # Update direction arrows (2D only)
        # Update direction arrows (2D only)
        if self.plot_type == '2d' and self.robot1_arrow is not None and self.robot2_arrow is not None:
            # Get yaw angles if available
            if 'robot1_yaw' in current_data.columns:
                r1_yaw = current_data['robot1_yaw'].iloc[0]
            elif 'ep03_yaw' in current_data.columns:
                r1_yaw = current_data['ep03_yaw'].iloc[0]
            else:
                # If no yaw is available, calculate direction from previous position
                if len(self.robot_trails['robot_1']) >= 2:
                    prev_x, prev_y = self.robot_trails['robot_1'][-2]
                    r1_yaw = np.arctan2(r1_pos_y - prev_y, r1_pos_x - prev_x)
                else:
                    r1_yaw = 0
                    
            if 'robot2_yaw' in current_data.columns:
                r2_yaw = current_data['robot2_yaw'].iloc[0]
            elif 'ep05_yaw' in current_data.columns:
                r2_yaw = current_data['ep05_yaw'].iloc[0]
            else:
                # If no yaw is available, calculate direction from previous position
                if len(self.robot_trails['robot_2']) >= 2:
                    prev_x, prev_y = self.robot_trails['robot_2'][-2]
                    r2_yaw = np.arctan2(r2_pos_y - prev_y, r2_pos_x - prev_x)
                else:
                    r2_yaw = 0
            
            # Calculate arrow direction from yaw
            arrow_len = 0.4
            r1_dx = arrow_len * np.cos(r1_yaw)
            r1_dy = arrow_len * np.sin(r1_yaw)
            r2_dx = arrow_len * np.cos(r2_yaw)
            r2_dy = arrow_len * np.sin(r2_yaw)
            
            # Update arrows
            self.robot1_arrow.set_offsets(np.array([[r1_pos_x, r1_pos_y]]))
            self.robot1_arrow.set_UVC(r1_dx, r1_dy)
            
            self.robot2_arrow.set_offsets(np.array([[r2_pos_x, r2_pos_y]]))
            self.robot2_arrow.set_UVC(r2_dx, r2_dy)
            
        # Update position text
        if hasattr(self, 'robot1_text') and self.robot1_text is not None:
            self.robot1_text.set_position((r1_pos_x + 0.3, r1_pos_y + 0.3))
            self.robot1_text.set_text(f"Robot 1\nX: {r1_pos_x:.2f}\nY: {r1_pos_y:.2f}")
            
        if hasattr(self, 'robot2_text') and self.robot2_text is not None:
            self.robot2_text.set_position((r2_pos_x + 0.3, r2_pos_y + 0.3))
            self.robot2_text.set_text(f"Robot 2\nX: {r2_pos_x:.2f}\nY: {r2_pos_y:.2f}")
        
        # Update point clouds if they should be shown
        if self.show_pcl:
            # Check if we have point cloud data
            pcl_cols = ['robot1_pcl_x', 'robot1_pcl_y', 'robot1_pcl_z',
                      'robot2_pcl_x', 'robot2_pcl_y', 'robot2_pcl_z']
            
            if all(col in current_data.columns for col in pcl_cols[:2]):
                # Get radar point cloud data for robot 1
                r1_radar_x = current_data['robot1_pcl_x'].values
                r1_radar_y = current_data['robot1_pcl_y'].values
                r1_radar_z = current_data['robot1_pcl_z'].values if 'robot1_pcl_z' in current_data.columns else np.zeros_like(r1_radar_x)
                
                # Update robot 1 point cloud
                if self.plot_type == '3d':
                    self.robot1_pcl._offsets3d = (r1_radar_x, r1_radar_y, r1_radar_z)
                else:
                    self.robot1_pcl.set_offsets(np.column_stack([r1_radar_x, r1_radar_y]))
            
            if all(col in current_data.columns for col in pcl_cols[3:5]):
                # Get radar point cloud data for robot 2
                r2_radar_x = current_data['robot2_pcl_x'].values
                r2_radar_y = current_data['robot2_pcl_y'].values
                r2_radar_z = current_data['robot2_pcl_z'].values if 'robot2_pcl_z' in current_data.columns else np.zeros_like(r2_radar_x)
                
                # Update robot 2 point cloud
                if self.plot_type == '3d':
                    self.robot2_pcl._offsets3d = (r2_radar_x, r2_radar_y, r2_radar_z)
                else:
                    self.robot2_pcl.set_offsets(np.column_stack([r2_radar_x, r2_radar_y]))
        else:
            # Hide point clouds
            if self.plot_type == '3d':
                self.robot1_pcl._offsets3d = ([], [], [])
                self.robot2_pcl._offsets3d = ([], [], [])
            else:
                self.robot1_pcl.set_offsets(np.zeros((0, 2)))
                self.robot2_pcl.set_offsets(np.zeros((0, 2)))
        
        # Update robot trails (2D only)
        if self.plot_type == '2d' and self.robot1_trail is not None and self.robot2_trail is not None:
            # Add new position to trails
            self.robot_trails['robot_1'].append((r1_pos_x, r1_pos_y))
            self.robot_trails['robot_2'].append((r2_pos_x, r2_pos_y))
            
            # Limit trail length
            if len(self.robot_trails['robot_1']) > self.trail_length:
                self.robot_trails['robot_1'] = self.robot_trails['robot_1'][-self.trail_length:]
            if len(self.robot_trails['robot_2']) > self.trail_length:
                self.robot_trails['robot_2'] = self.robot_trails['robot_2'][-self.trail_length:]
            
            # Update trail plots if trails should be shown
            if self.show_trails:
                if self.robot_trails['robot_1']:
                    x1, y1 = zip(*self.robot_trails['robot_1'])
                    self.robot1_trail.set_data(x1, y1)
                else:
                    self.robot1_trail.set_data([], [])
                    
                if self.robot_trails['robot_2']:
                    x2, y2 = zip(*self.robot_trails['robot_2'])
                    self.robot2_trail.set_data(x2, y2)
                else:
                    self.robot2_trail.set_data([], [])
            else:
                # Hide trails
                self.robot1_trail.set_data([], [])
                self.robot2_trail.set_data([], [])
    
    def update_vicon_visualization(self):
        """Update visualization for Vicon data"""
        if not self.frames:
            return
        
        # Get frame data
        frame = self.frames[self.current_frame]
        robots = frame['robots']
        timestamp = frame['timestamp']
        trails = frame['trails']
        
        # Update time display (relative to first frame)
        rel_time = timestamp - self.frames[0]['timestamp']
        self.time_text.set_text(f"Time: {rel_time:.2f}s")
        
        # Map robot names to our standard robot_1/robot_2 naming
        robot_name_map = {'ep03': 'robot_1', 'ep05': 'robot_2'}
        robot_pos_map = {}
        
        for robot_name, data in robots.items():
            std_name = robot_name_map.get(robot_name)
            if std_name:
                robot_pos_map[std_name] = {
                    'x': data['x'],
                    'y': data['y'],
                    'z': data['z'],
                    'yaw': data['yaw']
                }
        
        # Update robot positions
        for std_name, pos in robot_pos_map.items():
            x, y, z = pos['x'], pos['y'], pos['z']
            yaw = pos['yaw']
            
            if std_name == 'robot_1':
                if self.plot_type == '3d':
                    self.robot1_pos._offsets3d = ([x], [y], [z])
                else:
                    self.robot1_pos.set_offsets(np.column_stack([x, y]))
                    
                # Update direction arrow (2D only)
                # Update direction arrow (2D only)
                if self.plot_type == '2d' and self.robot1_arrow is not None:
                    arrow_len = 0.4
                    dx = arrow_len * np.cos(yaw)
                    dy = arrow_len * np.sin(yaw)
                    self.robot1_arrow.set_offsets([[x, y]])
                    self.robot1_arrow.set_UVC([dx], [dy])
                
                # Update position text
                if hasattr(self, 'robot1_text') and self.robot1_text is not None:
                    self.robot1_text.set_position((x + 0.3, y + 0.3))
                    self.robot1_text.set_text(f"Robot 1\nX: {x:.2f}\nY: {y:.2f}")
                
                # Update trail
                if self.plot_type == '2d' and self.robot1_trail is not None and self.show_trails:
                    vicon_name = [k for k, v in robot_name_map.items() if v == std_name][0]
                    if vicon_name in trails and trails[vicon_name]:
                        trail_x, trail_y = zip(*trails[vicon_name])
                        self.robot1_trail.set_data(trail_x, trail_y)
                    else:
                        self.robot1_trail.set_data([], [])
            
            elif std_name == 'robot_2':
                if self.plot_type == '3d':
                    self.robot2_pos._offsets3d = ([x], [y], [z])
                else:
                    self.robot2_pos.set_offsets(np.column_stack([x, y]))
                    
                # Update direction arrow (2D only)
                if self.plot_type == '2d' and self.robot2_arrow is not None:
                    arrow_len = 0.4
                    dx = arrow_len * np.cos(yaw)
                    dy = arrow_len * np.sin(yaw)
                    self.robot2_arrow.set_offsets([[x, y]])
                    self.robot2_arrow.set_UVC([dx], [dy])
                
                # Update position text
                if hasattr(self, 'robot2_text') and self.robot2_text is not None:
                    self.robot2_text.set_position((x + 0.3, y + 0.3))
                    self.robot2_text.set_text(f"Robot 2\nX: {x:.2f}\nY: {y:.2f}")
                
                # Update trail
                if self.plot_type == '2d' and self.robot2_trail is not None and self.show_trails:
                    vicon_name = [k for k, v in robot_name_map.items() if v == std_name][0]
                    if vicon_name in trails and trails[vicon_name]:
                        trail_x, trail_y = zip(*trails[vicon_name])
                        self.robot2_trail.set_data(trail_x, trail_y)
                    else:
                        self.robot2_trail.set_data([], [])
        
        # Hide robots that aren't in this frame
        if 'robot_1' not in robot_pos_map:
            if self.plot_type == '3d':
                self.robot1_pos._offsets3d = ([], [], [])
            else:
                self.robot1_pos.set_offsets([])
                if self.robot1_arrow is not None:
                    self.robot1_arrow.set_offsets([])
                    self.robot1_arrow.set_UVC([], [])
                if self.robot1_trail is not None:
                    self.robot1_trail.set_data([], [])
            if hasattr(self, 'robot1_text') and self.robot1_text is not None:
                self.robot1_text.set_text('')
        
        if 'robot_2' not in robot_pos_map:
            if self.plot_type == '3d':
                self.robot2_pos._offsets3d = ([], [], [])
            else:
                self.robot2_pos.set_offsets([])
                if self.robot2_arrow is not None:
                    self.robot2_arrow.set_offsets([])
                    self.robot2_arrow.set_UVC([], [])
                if self.robot2_trail is not None:
                    self.robot2_trail.set_data([], [])
            if hasattr(self, 'robot2_text') and self.robot2_text is not None:
                self.robot2_text.set_text('')
    
    def update_time(self, val):
        """Callback for the time slider."""
        frame_idx = int(val)
        self.update_visualization(frame_idx)
        self.fig.canvas.draw_idle()
    
    def toggle_play(self, event):
        """Toggle animation playback."""
        self.playing = not self.playing
        
        if self.playing:
            self.play_button.label.set_text('Pause')
            self.animate()
        else:
            self.play_button.label.set_text('Play')
    
    def animate(self):
        """Animation loop."""
        if not self.playing:
            return
        
        # Get max frame index
        max_frame = len(self.timestamps) - 1 if hasattr(self, 'timestamps') and self.timestamps is not None else self.frame_count - 1
        
        # If we're at the end, loop back to start
        if self.current_frame >= max_frame:
            self.current_frame = 0
            self.slider.set_val(self.current_frame)
            return
        
        # Calculate current and next timestamps
        if hasattr(self, 'timestamps') and self.timestamps is not None:
            current_time = self.timestamps[self.current_frame]
            next_frame = self.current_frame + 1
            next_time = self.timestamps[next_frame]
            time_diff = (next_time - current_time) / self.animation_speed
        elif self.frames:
            current_time = self.frames[self.current_frame]['timestamp']
            next_frame = self.current_frame + 1
            next_time = self.frames[next_frame]['timestamp']
            time_diff = (next_time - current_time) / self.animation_speed
        else:
            # Fallback to constant frame rate
            time_diff = 0.1 / self.animation_speed
            next_frame = self.current_frame + 1
        
        # Limit frame time for stability
        time_diff = max(0.01, min(time_diff, 0.5))
        
        # Advance frame
        self.current_frame = next_frame
        
        # Update slider (this will trigger visualization update)
        self.slider.set_val(self.current_frame)
        
        # Schedule next frame update
        if self.playing:
            self.fig.canvas.start_event_loop(time_diff)
            self.fig.canvas.mpl_connect('draw_event', 
                                      lambda event: self.animate() if self.playing else None)
    
    def decrease_speed(self, event):
        """Decrease animation speed."""
        self.animation_speed = max(0.25, self.animation_speed / 1.5)
        print(f"Animation speed: {self.animation_speed:.1f}x")
    
    def increase_speed(self, event):
        """Increase animation speed."""
        self.animation_speed = min(4.0, self.animation_speed * 1.5)
        print(f"Animation speed: {self.animation_speed:.1f}x")
    
    def toggle_view(self, event):
        """Toggle between 2D and 3D views."""
        # Switch view mode
        self.plot_type = '3d' if self.plot_type == '2d' else '2d'
        self.view_button.label.set_text(f'View: {self.plot_type.upper()}')
        
        # Clear the figure and recreate the visualization
        plt.clf()
        self.setup_visualization()
        self.update_visualization(self.current_frame)
        self.fig.canvas.draw_idle()
    
    def toggle_pcl(self, event):
        """Toggle point cloud visibility."""
        self.show_pcl = not self.show_pcl
        self.pcl_button.label.set_text('PCL: ON' if self.show_pcl else 'PCL: OFF')
        self.update_visualization(self.current_frame)
        self.fig.canvas.draw_idle()
    
    def toggle_trails(self, event):
        """Toggle trail visibility."""
        self.show_trails = not self.show_trails
        self.trail_button.label.set_text('Trails: ON' if self.show_trails else 'Trails: OFF')
        self.update_visualization(self.current_frame)
        self.fig.canvas.draw_idle()
    
    def run_visualization(self):
        """Run the complete visualization."""
        if self.data is None and not self.frames:
            print("Error: No data loaded")
            return False
        
        # Set up visualization
        self.setup_visualization()
        
        # Update with initial frame
        self.update_visualization(0)
        
        # Connect keyboard shortcuts
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Show the visualization
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)  # Make room for controls
        plt.show()
        return True
    
    def on_key_press(self, event):
        """Handle keyboard shortcuts."""
        if event.key == ' ':  # Space bar
            self.toggle_play(event)
        elif event.key == 'right':  # Right arrow
            max_frame = len(self.timestamps) - 1 if hasattr(self, 'timestamps') and self.timestamps is not None else self.frame_count - 1
            self.current_frame = min(self.current_frame + 1, max_frame)
            self.slider.set_val(self.current_frame)
        elif event.key == 'left':  # Left arrow
            self.current_frame = max(0, self.current_frame - 1)
            self.slider.set_val(self.current_frame)
        elif event.key == 'v':  # 'v' key
            self.toggle_view(event)
        elif event.key == 'p':  # 'p' key
            self.toggle_pcl(event)
        elif event.key == 't':  # 't' key
            self.toggle_trails(event)


def find_transformed_files(base_path, scenario=None):
    """
    Find all transformed CSV files in the given directory structure.
    
    Args:
        base_path (str): Path to the transformed data directory
        scenario (str, optional): Filter for a specific scenario
        
    Returns:
        list: List of dictionaries with file information
    """
    results = []
    
    try:
        # If we have a specific scenario
        if scenario:
            scenario_path = os.path.join(base_path, scenario)
            if not os.path.exists(scenario_path):
                print(f"Error: Scenario directory not found: {scenario_path}")
                return results
            
            # Search for transformed files in this scenario directory
            for dataset_dir in os.listdir(scenario_path):
                dataset_path = os.path.join(scenario_path, dataset_dir)
                if os.path.isdir(dataset_path):
                    for file in os.listdir(dataset_path):
                        if file.startswith("transformed_") and file.endswith(".csv"):
                            results.append({
                                'path': os.path.join(dataset_path, file),
                                'scenario': scenario,
                                'dataset': dataset_dir,
                                'file': file
                            })
        else:
            # Search all scenarios
            for scenario in os.listdir(base_path):
                scenario_path = os.path.join(base_path, scenario)
                if os.path.isdir(scenario_path):
                    for dataset_dir in os.listdir(scenario_path):
                        dataset_path = os.path.join(scenario_path, dataset_dir)
                        if os.path.isdir(dataset_path):
                            for file in os.listdir(dataset_path):
                                if file.startswith("transformed_") and file.endswith(".csv"):
                                    results.append({
                                        'path': os.path.join(dataset_path, file),
                                        'scenario': scenario,
                                        'dataset': dataset_dir,
                                        'file': file
                                    })
    except Exception as e:
        print(f"Error searching for transformed files: {e}")
    
    return results


def find_vicon_files(base_path, scenario=None):
    """
    Find all Vicon data files in the given directory structure.
    
    Args:
        base_path (str): Path to the Vicon data directory
        scenario (str, optional): Filter for a specific scenario
        
    Returns:
        list: List of dictionaries with file information
    """
    results = []
    
    try:
        # If we have a specific scenario
        if scenario:
            scenario_path = os.path.join(base_path, scenario)
            if not os.path.exists(scenario_path):
                print(f"Error: Scenario directory not found: {scenario_path}")
                return results
            
            # Search for Vicon files in this scenario directory
            for file in os.listdir(scenario_path):
                if file.endswith(".txt") and not file.endswith("_robot_positions.txt"):
                    results.append({
                        'path': os.path.join(scenario_path, file),
                        'scenario': scenario,
                        'file': file
                    })
        else:
            # Search all scenarios
            for scenario in os.listdir(base_path):
                scenario_path = os.path.join(base_path, scenario)
                if os.path.isdir(scenario_path):
                    for file in os.listdir(scenario_path):
                        if file.endswith(".txt") and not file.endswith("_robot_positions.txt"):
                            results.append({
                                'path': os.path.join(scenario_path, file),
                                'scenario': scenario,
                                'file': file
                            })
    except Exception as e:
        print(f"Error searching for Vicon files: {e}")
    
    return results


def main():
    """Main function to run the visualization."""
    parser = argparse.ArgumentParser(description='Interactive RoboFUSE Visualization')
    parser.add_argument('--input', '-i', type=str, default=None,
                        help='Path to transformed data CSV file or Vicon TXT file')
    parser.add_argument('--transformed-dir', '-t', type=str, 
                        default='/media/yugi/MAIN DRIVE/RoboFUSE_Dataset/Transformed_Data',
                        help='Base directory containing transformed data')
    parser.add_argument('--vicon-dir', '-v', type=str, 
                        default='/media/yugi/MAIN DRIVE/RoboFUSE_Dataset/vicon_data',
                        help='Base directory containing vicon data')
    parser.add_argument('--scenario', '-s', type=str, default=None,
                        help='Specific scenario to visualize (e.g., CPPS_Horizontal)')
    parser.add_argument('--plot-type', '-p', type=str, choices=['2d', '3d'], 
                        default='2d', help='Initial visualization type (default: 2d)')
    parser.add_argument('--data-type', '-d', type=str, choices=['transformed', 'vicon', 'auto'], 
                        default='auto', help='Type of data to visualize (default: auto-detect)')
    
    args = parser.parse_args()
    
    # If a specific input file is provided, use it
    if args.input:
        data_path = args.input
        data_type = args.data_type
        
        # Auto-detect data type if not specified
        if data_type == 'auto':
            if data_path.endswith('.csv'):
                data_type = 'transformed'
            elif data_path.endswith('.txt'):
                data_type = 'vicon'
            else:
                print(f"Error: Could not determine data type for file {data_path}")
                return 1
        
        # Create visualizer and run
        visualizer = RoboFUSEVisualizer(data_path)
        visualizer.plot_type = args.plot_type
        
        if visualizer.load_data():
            visualizer.run_visualization()
        else:
            print(f"Error: Could not load data from {data_path}")
            return 1
    else:
        # Determine data type if not specified
        data_type = args.data_type
        
        if data_type == 'auto' or data_type == 'transformed':
            # Try to find transformed data first
            files = find_transformed_files(args.transformed_dir, args.scenario)
            
            if files:
                data_type = 'transformed'
            elif data_type == 'auto':
                # If no transformed files found and auto-detect is on, try Vicon files
                data_type = 'vicon'
                files = find_vicon_files(args.vicon_dir, args.scenario)
            else:
                print(f"Error: No transformed data files found in {args.transformed_dir}")
                return 1
        elif data_type == 'vicon':
            # Find Vicon files
            files = find_vicon_files(args.vicon_dir, args.scenario)
        
        if not files:
            print(f"Error: No {data_type} data files found")
            if args.scenario:
                print(f"Check if scenario '{args.scenario}' exists")
            return 1
        
        # If there's only one file, use it
        if len(files) == 1:
            data_path = files[0]['path']
            print(f"Using the only available {data_type} file: {data_path}")
            
            # Create visualizer and run
            visualizer = RoboFUSEVisualizer(data_path)
            visualizer.plot_type = args.plot_type
            
            if visualizer.load_data():
                visualizer.run_visualization()
            else:
                print(f"Error: Could not load data from {data_path}")
                return 1
        else:
            # Let user choose from available files
            print(f"\nAvailable {data_type} data files:")
            for i, file_info in enumerate(files):
                if data_type == 'transformed':
                    print(f"[{i}] {file_info['scenario']} - {file_info['dataset']}")
                else:
                    print(f"[{i}] {file_info['scenario']} - {file_info['file']}")
            
            try:
                choice = input("\nEnter file number to visualize: ")
                file_idx = int(choice)
                if 0 <= file_idx < len(files):
                    data_path = files[file_idx]['path']
                    
                    # Create visualizer and run
                    visualizer = RoboFUSEVisualizer(data_path)
                    visualizer.plot_type = args.plot_type
                    
                    if visualizer.load_data():
                        visualizer.run_visualization()
                    else:
                        print(f"Error: Could not load data from {data_path}")
                        return 1
                else:
                    print(f"Error: Invalid choice. Must be between 0 and {len(files)-1}")
                    return 1
            except ValueError:
                print("Error: Please enter a valid number")
                return 1
            except KeyboardInterrupt:
                print("\nVisualization canceled by user")
                return 0
    
    return 0


if __name__ == "__main__":
    main()
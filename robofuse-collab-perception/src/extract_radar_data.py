#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import pandas as pd
import os
import struct
from datetime import datetime
import subprocess
import threading
import pathlib
import json
import tempfile
from typing import Dict, List, Any
import re
import sqlite3
import glob

from sensor_msgs.msg import PointCloud2

class RadarDataExtractor(Node):
    def __init__(self, robot_id='ep03', output_dir=None, scenario_name=None, bag_timestamps=None):
        super().__init__('radar_data_extractor')
        
        self.robot_id = robot_id
        self.scenario_name = scenario_name or "unknown_scenario"
        self.output_dir = output_dir or os.path.expanduser(f'~/radar_data/{self.scenario_name}')
        
        self.radar_data = []  # Stores metadata of radar messages
        self.radar_points = []  # Stores extracted radar points
        self.bag_timestamps = bag_timestamps or {}  # Timestamps from the bag file
        self.message_count = 0  # Track message index for timestamp lookup
        
        self.radar_sub = self.create_subscription(
            PointCloud2,
            f'/{robot_id}/ti_mmwave/radar_scan_pcl',
            self.radar_callback,
            10
        )
        
        self.get_logger().info(f'Starting radar extraction for {robot_id}, scenario: {self.scenario_name}')
        self.get_logger().info(f'Data will be saved to {self.output_dir}')
        
        self.msg_count = 0
        self.start_time = datetime.now().timestamp()

    def extract_xyz_from_pointcloud2(self, cloud_msg):
        """Extract XYZ points from a PointCloud2 message"""
        points = []
        x_offset = y_offset = z_offset = None
        
        for field in cloud_msg.fields:
            if (field.name == 'x'):
                x_offset = field.offset
            elif (field.name == 'y'):
                y_offset = field.offset
            elif (field.name == 'z'):
                z_offset = field.offset
        
        if None in (x_offset, y_offset, z_offset):
            return points
        
        point_step = cloud_msg.point_step
        
        for i in range(0, len(cloud_msg.data), point_step):
            try:
                x = struct.unpack_from('f', cloud_msg.data, i + x_offset)[0]
                y = struct.unpack_from('f', cloud_msg.data, i + y_offset)[0]
                z = struct.unpack_from('f', cloud_msg.data, i + z_offset)[0]
                points.append((x, y, z))
            except Exception:
                pass
        
        return points

    def radar_callback(self, msg):
        """Process incoming radar point cloud messages"""
        # Use the message's header timestamp
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    
        # Create a unique message ID
        msg_id = f"{msg.header.stamp.sec}.{msg.header.stamp.nanosec}"
    
        # Look up timestamp by message index
        bag_timestamp = self.bag_timestamps.get(self.message_count, timestamp)
         
        pcl_info = {
            
            'bag_timestamp': bag_timestamp,  # Use the lookup by index
            'frame_id': msg.header.frame_id,
            'height': msg.height,
            'width': msg.width,
            'point_count': msg.width * msg.height,
            'robot_id': self.robot_id,
            'scenario': self.scenario_name,
            
        }
        
        self.radar_data.append(pcl_info)
        points = self.extract_xyz_from_pointcloud2(msg)
        
        for pt in points:
            self.radar_points.append({
                
                'bag_timestamp': bag_timestamp,
                'x': pt[0],
                'y': pt[1],
                'z': pt[2],
                'robot_id': self.robot_id,
                'scenario': self.scenario_name,
                
            })
        
        self.msg_count += 1
        self.message_count += 1  # Increment counter for timestamp lookup
        if self.msg_count % 50 == 0:
            elapsed = datetime.now().timestamp() - self.start_time
            rate = self.msg_count / elapsed if elapsed > 0 else 0
            self.get_logger().info(f'Processed {self.msg_count} messages, {len(self.radar_points)} points ({rate:.1f} msgs/sec)')

    def save_data(self):
        """Save extracted data to CSV files"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        if self.radar_data:
            pd.DataFrame(self.radar_data).to_csv(os.path.join(self.output_dir, f'{self.robot_id}_radar_metadata.csv'), index=False)
        
        if self.radar_points:
            pd.DataFrame(self.radar_points).to_csv(os.path.join(self.output_dir, f'{self.robot_id}_radar_points.csv'), index=False)

        # Save the bag timestamps as well
        if self.bag_timestamps:
            with open(os.path.join(self.output_dir, f'{self.robot_id}_bag_timestamps.json'), 'w') as f:
                json.dump(self.bag_timestamps, f, indent=2)


def extract_timestamps_from_db3(db3_file, topic_name):
    """
    Extract timestamps directly from SQLite database for a specific topic
    
    Args:
        db3_file: Path to the .db3 file
        topic_name: Name of the topic to extract timestamps for
        
    Returns:
        Dictionary mapping message indices to timestamps
    """
    timestamps = {}
    
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db3_file)
        cursor = conn.cursor()
        
        # Get the topic ID for the radar topic
        cursor.execute("SELECT id FROM topics WHERE name = ?;", (topic_name,))
        topic_result = cursor.fetchone()
        
        if not topic_result:
            print(f"Topic {topic_name} not found in {db3_file}")
            conn.close()
            return timestamps
            
        topic_id = topic_result[0]
        
        # Get all message timestamps for this topic
        try:
            # Try the 'messages' table first (newer format)
            cursor.execute("SELECT timestamp FROM messages WHERE topic_id = ? ORDER BY timestamp;", (topic_id,))
            message_results = cursor.fetchall()
        except:
            try:
                # Try the 'message' table (older format)
                cursor.execute("SELECT timestamp FROM message WHERE topic_id = ? ORDER BY timestamp;", (topic_id,))
                message_results = cursor.fetchall()
            except:
                print(f"Could not query message table in {db3_file}")
                conn.close()
                return timestamps
        
        # Process results
        for i, (timestamp,) in enumerate(message_results):
            # Convert nanoseconds to seconds
            timestamps[i] = timestamp / 1e9
            
        conn.close()
        print(f"Extracted {len(timestamps)} timestamps for {topic_name} from {db3_file}")
        return timestamps
        
    except Exception as e:
        print(f"Error extracting timestamps from {db3_file}: {e}")
        return {}


def process_rosbag(rosbag_path, robot_id, output_dir, scenario_name):
    """Process a single rosbag file"""
    # Find the db3 file
    db3_files = glob.glob(os.path.join(rosbag_path, "*.db3"))
    if not db3_files:
        print(f"No .db3 files found in {rosbag_path}")
        return
        
    db3_file = db3_files[0]
    topic_name = f'/{robot_id}/ti_mmwave/radar_scan_pcl'
    
    print(f"Extracting timestamps for {topic_name} from {db3_file}...")
    bag_timestamps = extract_timestamps_from_db3(db3_file, topic_name)
    
    if not bag_timestamps:
        print(f"No timestamps found for {topic_name} in {db3_file}")
        return
    
    # Continue with the rest of your processing
    print(f"Playing bag and extracting radar data...")
    play_process = subprocess.Popen(['ros2', 'bag', 'play', rosbag_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    rclpy.init()
    extractor = RadarDataExtractor(robot_id, output_dir, scenario_name, bag_timestamps)
    
    def monitor_rosbag():
        play_process.wait()
        rclpy.shutdown()
    
    monitor_thread = threading.Thread(target=monitor_rosbag, daemon=True)
    monitor_thread.start()
    
    try:
        rclpy.spin(extractor)
    except KeyboardInterrupt:
        pass
    finally:
        extractor.save_data()
        extractor.destroy_node()
        if play_process.poll() is None:
            play_process.terminate()
            play_process.wait()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except RuntimeError:
            print("ROS context already shut down")


def extract_all_radar_data(dataset_root):
    """Process all rosbags in the dataset"""
    rosbag_dirs = []
    scenario_types = ['CPPS_Diagonal', 'CPPS_Horizontal', 
    'CPPS_Vertical',
    'CPPS_Diagonal_Horizontal',
    'CPPS_Horizontal_Diagonal',
    'CPPS_Horizontal_Vertical',
    'CPPS_Vertical_Horizontal']
    
    for scenario in scenario_types:
        scenario_path = pathlib.Path(dataset_root) / scenario
        if not scenario_path.exists():
            continue
        
        for robot_type in ['Robot_1', 'Robot_2']:
            rosbag_path = scenario_path / robot_type / 'rosbag'
            if not rosbag_path.exists():
                continue
            
            for item in rosbag_path.iterdir():
                if item.is_dir() and any(f.suffix == '.db3' for f in item.iterdir()):
                    rosbag_dirs.append({'path': str(item), 'scenario': scenario, 'robot_type': robot_type})
    
    print(f"Found {len(rosbag_dirs)} rosbag directories to process")
    
    for idx, rosbag_info in enumerate(rosbag_dirs):
        rosbag_path, scenario, robot_type = rosbag_info.values()
        robot_id = 'ep03' if robot_type == 'Robot_1' else 'ep05' if robot_type == 'Robot_2' else 'cable_robot'
        output_dir = os.path.join('/media/yugi/MAIN DRIVE/RoboFUSE_Dataset/Extracted_Radar_Data', 
                                scenario, robot_type, os.path.basename(rosbag_path))
        
        print(f"\nProcessing {idx+1}/{len(rosbag_dirs)}: {rosbag_path}")
        process_rosbag(rosbag_path, robot_id, output_dir, scenario)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        process_rosbag(sys.argv[1], 
                      sys.argv[2] if len(sys.argv) > 2 else 'ep03', 
                      '/media/yugi/MAIN DRIVE/RoboFUSE_Dataset/ExtractedRadar_Data', 
                      "single_rosbag")
    else:
        extract_all_radar_data('/media/yugi/MAIN DRIVE/RoboFUSE_Dataset/CPPS_Static_Scenario')
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
import json

class RoboFUSEPreprocessor:
    """
    A class for preprocessing and synchronizing RoboFUSE dataset
    containing radar and Vicon motion capture data.
    """
    
    def __init__(self, base_path, scenario, output_dir="output"):
        """
        Initialize preprocessor with dataset paths.
        
        Args:
            base_path (str): Path to the extracted data directory
            scenario (str): Name of the scenario (e.g., "CPPS_Horizontal")
            output_dir (str): Directory to save output files
        """
        self.base_path = base_path
        self.scenario = scenario
        self.output_dir = output_dir
        
        # Setup paths
        self.radar_base_path = os.path.join(base_path, "Extracted_Radar_Data", scenario)
        self.vicon_base_path = os.path.join(base_path, "vicon_data", scenario)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.join(output_dir, scenario), exist_ok=True)
        
        # Store loaded data
        self.robot1_data = None
        self.robot2_data = None
        self.vicon_data = None
        self.resampled_vicon_data = None
        self.synchronized_data = None
        
        print(f"Initialized preprocessor for scenario: {scenario}")
        print(f"Radar data path: {self.radar_base_path}")
        print(f"Vicon data path: {self.vicon_base_path}")
    
    def find_matching_files(self, robot_id):
        """
        Find available radar data files for a specific robot.
        
        Args:
            robot_id (str): Robot identifier (Robot_1 or Robot_2)
            
        Returns:
            list: List of available data directories
        """
        robot_path = os.path.join(self.radar_base_path, robot_id)
        if not os.path.exists(robot_path):
            print(f"Warning: Path {robot_path} does not exist")
            return []
            
        # List directories that contain radar data
        directories = [d for d in os.listdir(robot_path) 
                      if os.path.isdir(os.path.join(robot_path, d))]
        
        available_dirs = []
        for directory in directories:
            dir_path = os.path.join(robot_path, directory)
            # Check if radar_points.csv exists
            if os.path.exists(os.path.join(dir_path, f"{'ep03' if robot_id=='Robot_1' else 'ep05'}_radar_points.csv")):
                available_dirs.append(directory)
        
        return available_dirs
    
    def find_vicon_files(self):
        """
        Find available Vicon data files for the scenario.
        
        Returns:
            list: List of available Vicon CSV files
        """
        if not os.path.exists(self.vicon_base_path):
            print(f"Warning: Vicon path {self.vicon_base_path} does not exist")
            return []
            
        vicon_files = [f for f in os.listdir(self.vicon_base_path) 
                      if f.endswith('.csv') and not f.endswith('_robot_positions.csv')]
        
        return vicon_files
    
    def print_available_data(self):
        """
        Print available data for the scenario to help with selection.
        """
        print(f"\nAvailable data for scenario {self.scenario}:")
        
        # Check Robot_1 data
        robot1_dirs = self.find_matching_files("Robot_1")
        print(f"Robot_1 data directories ({len(robot1_dirs)}):")
        for i, directory in enumerate(robot1_dirs):
            print(f"  [{i}] {directory}")
        
        # Check Robot_2 data
        robot2_dirs = self.find_matching_files("Robot_2")
        print(f"Robot_2 data directories ({len(robot2_dirs)}):")
        for i, directory in enumerate(robot2_dirs):
            print(f"  [{i}] {directory}")
        
        # Check Vicon data
        vicon_files = self.find_vicon_files()
        print(f"Vicon data files ({len(vicon_files)}):")
        for i, file in enumerate(vicon_files):
            print(f"  [{i}] {file}")
    
    def load_radar_data(self, robot_id, directory):
        """
        Load radar data for a specific robot and directory.
        
        Args:
            robot_id (str): Robot identifier (Robot_1 or Robot_2)
            directory (str): Directory name containing the data
            
        Returns:
            pd.DataFrame: Loaded radar data
        """
        robot_code = 'ep03' if robot_id == 'Robot_1' else 'ep05'
        file_path = os.path.join(self.radar_base_path, robot_id, directory, f"{robot_code}_radar_points.csv")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Radar data file not found: {file_path}")
        
        # Load radar data
        print(f"Loading radar data from {file_path}")
        df = pd.read_csv(file_path)
        
        # Show column names for debugging
        print(f"Columns in {robot_id} radar data: {df.columns.tolist()}")
        
        # Create a timestamp column identifier for the robot
        timestamp_col = f"{robot_id.lower()}_timestamp"
        
        # Check if 'timestamp' column exists, otherwise look for alternatives
        if 'timestamp' in df.columns:
            df.rename(columns={'timestamp': timestamp_col}, inplace=True)
        else:
            # Look for timestamp-like columns
            possible_timestamp_cols = [col for col in df.columns if 'time' in col.lower()]
            if possible_timestamp_cols:
                print(f"Using '{possible_timestamp_cols[0]}' as timestamp column for {robot_id}")
                df.rename(columns={possible_timestamp_cols[0]: timestamp_col}, inplace=True)
            else:
                # If no timestamp column found, create one based on index
                print(f"Warning: No timestamp column found in {robot_id} data, creating artificial timestamps")
                df[timestamp_col] = np.arange(len(df)) / 100.0  # Assume 100Hz data
        
        # Add robot identifier
        df['Node_ID'] = robot_id.lower()
        
        return df
    
    def load_vicon_data(self, file_name):
        """
        Load Vicon motion capture data.
        
        Args:
            file_name (str): Vicon data file name
            
        Returns:
            pd.DataFrame: Loaded Vicon data
        """
        file_path = os.path.join(self.vicon_base_path, file_name)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Vicon data file not found: {file_path}")
        
        print(f"Loading Vicon data from {file_path}")
        df = pd.read_csv(file_path)
        
        # Check if there's a timestamp column
        if 'timestamp' in df.columns:
            # Rename to avoid confusion
            df.rename(columns={'timestamp': 'vicon_timestamp'}, inplace=True)
        elif 'sec' in df.columns and 'nsec' in df.columns:
            # Convert ROS timestamp to seconds
            df['vicon_timestamp'] = df['sec'] + df['nsec'] / 1e9
        
        # If timestamp not found, try to extract from robot pose data
        if 'vicon_timestamp' not in df.columns:
            # Try to extract timestamp from object/robot columns
            timestamp_cols = [col for col in df.columns if 'time' in col.lower()]
            if timestamp_cols:
                df['vicon_timestamp'] = df[timestamp_cols[0]]
            else:
                # Create artificial timestamp based on row index
                print("Warning: No timestamp column found in Vicon data, creating artificial timestamps")
                df['vicon_timestamp'] = np.arange(len(df)) / 100.0  # Assume 100Hz
        
        return df
    
    def resample_vicon_data(self, vicon_df, resample_interval='25ms'):
        """
        Resample Vicon data to a fixed time interval.
        
        Args:
            vicon_df (pd.DataFrame): Vicon data frame
            resample_interval (str): Pandas-compatible resampling interval
            
        Returns:
            pd.DataFrame: Resampled Vicon data
        """
        if vicon_df is None or vicon_df.empty:
            print("Warning: Empty Vicon data, skipping resampling")
            return vicon_df
        
        print(f"Resampling Vicon data with interval {resample_interval}")
        
        # Convert timestamp to datetime for resampling
        vicon_df['time_resample'] = pd.to_datetime(vicon_df['vicon_timestamp'], unit='s')
        
        # Separate non-numerical and numerical columns
        non_numerical_cols = [col for col in vicon_df.columns if vicon_df[col].dtype == 'object']
        numerical_cols = [col for col in vicon_df.columns 
                         if col not in non_numerical_cols + ['time_resample']]
        
        # First group by the timestamp column to handle any duplicate timestamps
        df_vicon_non_num = vicon_df.groupby('time_resample')[non_numerical_cols].first()
        df_vicon_num = vicon_df.groupby('time_resample')[numerical_cols].mean()
        
        # Combine numerical and non-numerical data
        df_vicon_combined = pd.concat([df_vicon_num, df_vicon_non_num], axis=1).reset_index()
        
        # Resample based on time interval and forward-fill missing values
        df_vicon_combined = df_vicon_combined.set_index('time_resample').sort_index()
        df_vicon_resampled = df_vicon_combined.resample(resample_interval).ffill().dropna().reset_index()
        
        # Drop the time_resample column as we'll use vicon_timestamp
        df_vicon_resampled = df_vicon_resampled.drop('time_resample', axis=1)
        
        return df_vicon_resampled
    
    def filter_radar_by_vicon_time(self, radar_df, vicon_df):
        """
        Filter radar data to match the time range in Vicon data.
        
        Args:
            radar_df (pd.DataFrame): Radar data frame
            vicon_df (pd.DataFrame): Vicon data frame
            
        Returns:
            pd.DataFrame: Filtered radar data
        """
        if radar_df is None or radar_df.empty or vicon_df is None or vicon_df.empty:
            print("Warning: Empty data, skipping filtering")
            return radar_df
        
        # Get node ID from the first row
        node_id = radar_df['Node_ID'].iloc[0]
        timestamp_col = f"{node_id}_timestamp"
        
        # Verify timestamp column exists
        if timestamp_col not in radar_df.columns:
            print(f"Error: Timestamp column '{timestamp_col}' not found in radar data columns: {radar_df.columns.tolist()}")
            # Try to find alternative timestamp column
            possible_timestamp_cols = [col for col in radar_df.columns if 'time' in col.lower()]
            if possible_timestamp_cols:
                timestamp_col = possible_timestamp_cols[0]
                print(f"Using alternative timestamp column: {timestamp_col}")
            else:
                print("No timestamp column found, returning unfiltered data")
                return radar_df
        
        start_time = vicon_df['vicon_timestamp'].min()
        end_time = vicon_df['vicon_timestamp'].max()
        
        print(f"Filtering radar data to Vicon time range: {start_time:.2f} to {end_time:.2f}")
        
        filtered_df = radar_df[
            (radar_df[timestamp_col] >= start_time) & 
            (radar_df[timestamp_col] <= end_time)
        ]
        
        print(f"Filtered radar data from {len(radar_df)} to {len(filtered_df)} points")
        
        return filtered_df
    
    def detect_and_fill_gaps(self, df, timestamp_col, sampling_interval, 
                        fields_to_interpolate, interpolation_enabled=False,
                        max_gap_to_fill=10.0):  # Maximum gap in seconds to fill
        # Ensure DataFrame is sorted by timestamp
        df = df.sort_values(timestamp_col).reset_index(drop=True)
        
        # Detect gaps where the timestamp difference exceeds threshold
        df['time_diff'] = df[timestamp_col].diff()
        
        # Find rows where there are gaps
        large_gaps = df[(df['time_diff'] > 5) & (df['time_diff'] <= max_gap_to_fill)]
        if not large_gaps.empty:
            print(f"Found {len(large_gaps)} large gaps (>5s, ≤{max_gap_to_fill}s) to fill")
            
            complete_timestamps = []
            for i in range(len(df) - 1):
                t_now, t_next = df.loc[i, timestamp_col], df.loc[i + 1, timestamp_col]
                complete_timestamps.append(t_now)
                
                # Fill gaps by interpolating timestamps based on sampling_interval
                gap_length = t_next - t_now
                # If gap is extremely large, handle it differently
                if gap_length > max_gap_to_fill:
                    print(f"Very large gap detected: {t_now:.2f} to {t_next:.2f} ({gap_length:.2f}s)")
                    print(f"Skipping detailed filling and inserting sparse points only")
                    # Insert just a few points to maintain continuity without overwhelming memory
                    num_sparse_points = min(20, int(gap_length / sampling_interval))
                    missing_timestamps = np.linspace(t_now + sampling_interval, 
                                                    t_next - sampling_interval, 
                                                    num_sparse_points)
                    complete_timestamps.extend(missing_timestamps)
                    
            
            complete_timestamps.append(df[timestamp_col].iloc[-1])
            
            # Create a DataFrame with all timestamps
            complete_df = pd.DataFrame({timestamp_col: complete_timestamps})
            
            # Merge the original data into the complete timestamp range
            df = pd.merge(complete_df, df, on=timestamp_col, how='left')
            
            # Fill gaps with interpolation if enabled
            if interpolation_enabled:
                for field in fields_to_interpolate:
                    if field in df.columns:
                        df[field] = df[field].interpolate(method='linear', limit_direction='both')
        
        # Remove temporary columns and return
        df.drop(columns=['time_diff'], errors='ignore', inplace=True)
        
        return df
    
    def synchronize_radar_nodes(self, filtered_radar_dfs, sampling_interval=0.05):
        """
        Synchronize radar data from different nodes.
        
        Args:
            filtered_radar_dfs (dict): Dictionary of radar DataFrames by node
            sampling_interval (float): Sampling interval in seconds
            
        Returns:
            pd.DataFrame: Synchronized radar data
        """
        if 'robot_1' not in filtered_radar_dfs or 'robot_2' not in filtered_radar_dfs:
            print("Error: Both Robot_1 and Robot_2 data are required for synchronization")
            return None
        
        # Determine which robot has more data points - use that as reference
        robot1_size = len(filtered_radar_dfs['robot_1'])
        robot2_size = len(filtered_radar_dfs['robot_2'])
        
        print(f"Robot_1 data size: {robot1_size} rows")
        print(f"Robot_2 data size: {robot2_size} rows")
        
        # Use the robot with more data as reference
        if robot1_size >= robot2_size:
            reference_robot = 'robot_1'
            other_robot = 'robot_2'
            print(f"Using Robot_1 as reference (more data points)")
        else:
            reference_robot = 'robot_2'
            other_robot = 'robot_1'
            print(f"Using Robot_2 as reference (more data points)")
        
        reference_df = filtered_radar_dfs[reference_robot]
        other_df = filtered_radar_dfs[other_robot]
        
        # Check for timestamp columns
        reference_timestamp_col = None
        other_timestamp_col = None
        
        # Find timestamp column for reference robot
        for col in reference_df.columns:
            if 'time' in col.lower() or 'timestamp' in col.lower():
                reference_timestamp_col = col
                print(f"Using '{reference_timestamp_col}' as timestamp for {reference_robot}")
                break
        
        # Find timestamp column for other robot
        for col in other_df.columns:
            if 'time' in col.lower() or 'timestamp' in col.lower():
                other_timestamp_col = col
                print(f"Using '{other_timestamp_col}' as timestamp for {other_robot}")
                break
        
        if not reference_timestamp_col or not other_timestamp_col:
            print("Error: Could not find timestamp columns for synchronization")
            return None
        
        # Detect and handle gaps in both robots' data
        for node_id, timestamp_col in [(reference_robot, reference_timestamp_col), (other_robot, other_timestamp_col)]:
            if node_id in filtered_radar_dfs:
                filtered_radar_dfs[node_id] = self.detect_and_fill_gaps(
                    filtered_radar_dfs[node_id],
                    timestamp_col=timestamp_col,
                    sampling_interval=sampling_interval,
                    fields_to_interpolate=['x', 'y', 'z'],
                    interpolation_enabled=False
                )
        
        # After gap filling, update references
        reference_df = filtered_radar_dfs[reference_robot]
        other_df = filtered_radar_dfs[other_robot]
        
        synchronized_data = []
        
        # Loop through reference robot timestamps
        for _, reference_row in reference_df.iterrows():
            sync_row = {}
            
            # Find closest matching timestamp in other robot data
            closest_idx = (np.abs(other_df[other_timestamp_col] - reference_row[reference_timestamp_col])).idxmin()
            other_data = other_df.loc[closest_idx].drop('Node_ID', errors='ignore').to_dict()
            
            # Calculate time difference for monitoring
            time_diff = abs(other_data[other_timestamp_col] - reference_row[reference_timestamp_col])
            
            # Add data - start with timestamps
            sync_row['robot_1_timestamp'] = (reference_row[reference_timestamp_col] if reference_robot == 'robot_1' 
                                          else other_data[other_timestamp_col])
            
            sync_row['robot_2_timestamp'] = (reference_row[reference_timestamp_col] if reference_robot == 'robot_2' 
                                          else other_data[other_timestamp_col])
            
            # Add reference robot data first
            for key, value in reference_row.items():
                if key != reference_timestamp_col and key != 'Node_ID':
                    # Don't prefix columns that are already prefixed
                    if key.startswith(f'{reference_robot}_'):
                        sync_row[key] = value
                    else:
                        sync_row[f'{reference_robot}_{key}'] = value
            
            # Add other robot data
            for key, value in other_data.items():
                if key != other_timestamp_col and key != 'Node_ID':
                    # Don't prefix columns that are already prefixed
                    if key.startswith(f'{other_robot}_'):
                        sync_row[key] = value
                    else:
                        sync_row[f'{other_robot}_{key}'] = value
            
            # Only add rows where time difference is below a threshold (0.5 seconds)
            # This prevents matching data that's too far apart in time
            if time_diff <= 0.5:
                synchronized_data.append(sync_row)
        
        # Convert to DataFrame
        synchronized_df = pd.DataFrame(synchronized_data)
        
        # Reorder columns to ensure robot_1 comes first, followed by robot_2
        column_order = (
            [col for col in synchronized_df.columns if col == 'robot_1_timestamp'] +
            [col for col in synchronized_df.columns if col.startswith('robot_1_') and col != 'robot_1_timestamp'] +
            [col for col in synchronized_df.columns if col == 'robot_2_timestamp'] +
            [col for col in synchronized_df.columns if col.startswith('robot_2_') and col != 'robot_2_timestamp']
        )
        
        # Make sure we're only selecting columns that actually exist
        column_order = [col for col in column_order if col in synchronized_df.columns]
        
        synchronized_df = synchronized_df[column_order]
        
        print(f"Created synchronized radar data with {len(synchronized_df)} points")
        print(f"Time synchronization threshold: 0.5 seconds")
        print(f"Average time difference between robot readings: {np.mean(np.abs(synchronized_df['robot_1_timestamp'] - synchronized_df['robot_2_timestamp'])):.6f} seconds")
        
        return synchronized_df
    
    def synchronize_radar_vicon(self, synchronized_radar_df, vicon_df):
        """
        Synchronize radar and Vicon data.
        
        Args:
            synchronized_radar_df (pd.DataFrame): Synchronized radar data
            vicon_df (pd.DataFrame): Vicon data
            
        Returns:
            pd.DataFrame: Final synchronized data
        """
        if synchronized_radar_df is None or synchronized_radar_df.empty or vicon_df is None or vicon_df.empty:
            print("Error: Both synchronized radar and Vicon data are required")
            return None
        
        # Check if we have robot_2_timestamp column
        if 'robot_2_timestamp' not in synchronized_radar_df.columns:
            # Find any column that might have timestamp for robot 2
            timestamp_cols = [col for col in synchronized_radar_df.columns 
                             if 'time' in col.lower() and 'robot_2' in col.lower()]
            if timestamp_cols:
                robot_2_timestamp_col = timestamp_cols[0]
                print(f"Using '{robot_2_timestamp_col}' for synchronization with Vicon")
            else:
                # Try any timestamp columns
                timestamp_cols = [col for col in synchronized_radar_df.columns if 'time' in col.lower()]
                if timestamp_cols:
                    robot_2_timestamp_col = timestamp_cols[0]
                    print(f"Using '{robot_2_timestamp_col}' for synchronization with Vicon")
                else:
                    print("Error: No timestamp column found for synchronization with Vicon")
                    return None
        else:
            robot_2_timestamp_col = 'robot_2_timestamp'
        
        synchronized_data = []
        
        for _, radar_row in synchronized_radar_df.iterrows():
            # Find closest Vicon timestamp to robot timestamp
            closest_vicon_idx = (np.abs(vicon_df['vicon_timestamp'] - radar_row[robot_2_timestamp_col])).idxmin()
            closest_vicon_row = vicon_df.loc[closest_vicon_idx]
            
            # Combine data
            combined_row = {**closest_vicon_row.to_dict(), **radar_row.to_dict()}
            synchronized_data.append(combined_row)
        
        # Create a DataFrame from the synchronized data
        final_df = pd.DataFrame(synchronized_data)
        
        # Organize columns
        # First vicon timestamps
        vicon_columns = [col for col in final_df.columns if 'vicon_timestamp' in col]
        
        # Then robot timestamps
        robot_timestamp_columns = [col for col in final_df.columns 
                                  if ('timestamp' in col or 'time' in col.lower()) and col.startswith('robot_')]
        
        # Then other robot data
        robot_data_columns = [col for col in final_df.columns 
                             if col.startswith('robot_') and col not in robot_timestamp_columns]
        
        # Then remaining columns
        other_columns = [col for col in final_df.columns 
                        if col not in vicon_columns + robot_timestamp_columns + robot_data_columns]
        
        # Reorder columns
        column_order = vicon_columns + robot_timestamp_columns + robot_data_columns + other_columns
        
        # Make sure we're only selecting columns that actually exist
        column_order = [col for col in column_order if col in final_df.columns]
        
        final_df = final_df[column_order]
        
        print(f"Created final synchronized data with {len(final_df)} points")
        
        return final_df
    
    def process_data(self, robot1_dir, robot2_dir, vicon_file, 
                   resample_interval='25ms', save_results=True):
        """
        Process the data by loading, filtering, and synchronizing.
        
        Args:
            robot1_dir (str): Directory containing Robot_1 data
            robot2_dir (str): Directory containing Robot_2 data
            vicon_file (str): Vicon data file name
            resample_interval (str): Pandas-compatible resampling interval
            save_results (bool): Whether to save results to files
            
        Returns:
            dict: Dictionary containing all processed dataframes
        """
        print(f"\nProcessing data for scenario {self.scenario}")
        print(f"Robot 1 directory: {robot1_dir}")
        print(f"Robot 2 directory: {robot2_dir}")
        print(f"Vicon file: {vicon_file}")
        
        # Step 1: Load radar data
        try:
            self.robot1_data = self.load_radar_data('Robot_1', robot1_dir)
            self.robot2_data = self.load_radar_data('Robot_2', robot2_dir)
        except FileNotFoundError as e:
            print(f"Error loading radar data: {e}")
            return None
        
        # Step 2: Load Vicon data
        try:
            self.vicon_data = self.load_vicon_data(vicon_file)
        except FileNotFoundError as e:
            print(f"Error loading Vicon data: {e}")
            return None
        
        # Step 3: Detect movement start
        print("\nAnalyzing robot movement patterns...")
        
        # Find position columns for movement detection
        robot1_pos_cols = []
        for suffix in ['_x', '_y', '_z']:
            for col in self.robot1_data.columns:
                if col.endswith(suffix):
                    robot1_pos_cols.append(col)
                    break
        
        robot2_pos_cols = []
        for suffix in ['_x', '_y', '_z']:
            for col in self.robot2_data.columns:
                if col.endswith(suffix):
                    robot2_pos_cols.append(col)
                    break
        
        # Find timestamp columns
        robot1_timestamp_col = None
        for col in self.robot1_data.columns:
            if 'time' in col.lower() or 'timestamp' in col.lower():
                robot1_timestamp_col = col
                break
        
        robot2_timestamp_col = None
        for col in self.robot2_data.columns:
            if 'time' in col.lower() or 'timestamp' in col.lower():
                robot2_timestamp_col = col
                break
        
        # Detect movement start for both robots
        robot1_start_time = None
        if robot1_pos_cols and robot1_timestamp_col:
            print("Detecting Robot_1 movement start...")
            robot1_start_time = self.detect_movement_start(
                self.robot1_data, robot1_pos_cols, robot1_timestamp_col
            )
        
        robot2_start_time = None
        if robot2_pos_cols and robot2_timestamp_col:
            print("Detecting Robot_2 movement start...")
            robot2_start_time = self.detect_movement_start(
                self.robot2_data, robot2_pos_cols, robot2_timestamp_col
            )
        
        # Use the latest start time as the common start time for data filtering
        if robot1_start_time and robot2_start_time:
            start_buffer = -2.0  # Start 2 seconds before movement to capture initial state
            common_start_time = max(robot1_start_time, robot2_start_time) + start_buffer
            print(f"Using common start time: {common_start_time:.2f} " +
                 f"(max of robot start times + {start_buffer:.1f}s buffer)")
            
            # Pre-filter radar data by time to remove data before robots start moving
            if robot1_timestamp_col:
                self.robot1_data = self.robot1_data[self.robot1_data[robot1_timestamp_col] >= common_start_time]
                print(f"Pre-filtered Robot_1 data to {len(self.robot1_data)} points starting from movement")
            
            if robot2_timestamp_col:
                self.robot2_data = self.robot2_data[self.robot2_data[robot2_timestamp_col] >= common_start_time]
                print(f"Pre-filtered Robot_2 data to {len(self.robot2_data)} points starting from movement")
            
            # Pre-filter Vicon data by time if it has a timestamp column
            if 'vicon_timestamp' in self.vicon_data.columns:
                # Apply a slightly larger buffer to Vicon to ensure we don't miss data
                vicon_start_time = common_start_time - 0.5
                self.vicon_data = self.vicon_data[self.vicon_data['vicon_timestamp'] >= vicon_start_time]
                print(f"Pre-filtered Vicon data to {len(self.vicon_data)} points starting from movement")
        
        # Step 4: Resample Vicon data
        self.resampled_vicon_data = self.resample_vicon_data(
            self.vicon_data, resample_interval=resample_interval
        )
        
        # Step 5: Filter radar data based on Vicon timestamps
        filtered_radar_dfs = {}
        filtered_radar_dfs['robot_1'] = self.filter_radar_by_vicon_time(
            self.robot1_data, self.resampled_vicon_data
        )
        filtered_radar_dfs['robot_2'] = self.filter_radar_by_vicon_time(
            self.robot2_data, self.resampled_vicon_data
        )
        
        # Step 6: Synchronize radar nodes
        self.synchronized_radar_data = self.synchronize_radar_nodes(filtered_radar_dfs)
        
        # Step 7: Synchronize radar and Vicon
        self.final_synchronized_data = self.synchronize_radar_vicon(
            self.synchronized_radar_data, self.resampled_vicon_data
        )
        
        # Save results if requested
        # Save results if requested
        if save_results:
            # Create a dataset-specific subfolder using the robot directories and vicon file
            dataset_folder = f"{robot1_dir}_{robot2_dir}_{os.path.splitext(vicon_file)[0]}"
            output_path = os.path.join(self.output_dir, self.scenario, dataset_folder)
            os.makedirs(output_path, exist_ok=True)
            
            if self.robot1_data is not None:
                self.robot1_data.to_csv(
                    os.path.join(output_path, f"robot1_raw_{robot1_dir}.csv"), index=False
                )
            
            if self.robot2_data is not None:
                self.robot2_data.to_csv(
                    os.path.join(output_path, f"robot2_raw_{robot2_dir}.csv"), index=False
                )
            
            if self.vicon_data is not None:
                self.vicon_data.to_csv(
                    os.path.join(output_path, f"vicon_raw_{os.path.splitext(vicon_file)[0]}.csv"), 
                    index=False
                )
            
            if self.resampled_vicon_data is not None:
                self.resampled_vicon_data.to_csv(
                    os.path.join(output_path, f"vicon_resampled_{os.path.splitext(vicon_file)[0]}.csv"), 
                    index=False
                )
            
            if self.synchronized_radar_data is not None:
                self.synchronized_radar_data.to_csv(
                    os.path.join(output_path, f"radar_synchronized_{robot1_dir}_{robot2_dir}.csv"), 
                    index=False
                )
            
            if self.final_synchronized_data is not None:
                filename = f"final_synchronized_{robot1_dir}_{robot2_dir}_{os.path.splitext(vicon_file)[0]}.csv"
                output_file = os.path.join(output_path, filename)
                self.final_synchronized_data.to_csv(output_file, index=False)
                print(f"Saved final synchronized data to: {output_file}")
            
            print(f"Saved processed data to {output_path}")
        
        return {
            'robot1_data': self.robot1_data,
            'robot2_data': self.robot2_data,
            'vicon_data': self.vicon_data,
            'resampled_vicon_data': self.resampled_vicon_data,
            'synchronized_radar_data': self.synchronized_radar_data,
            'final_synchronized_data': self.final_synchronized_data
        }
    


def main(process_all=False):
    """
    Main function to demonstrate preprocessing workflow.
    
    Args:
        process_all (bool): If True, automatically process all available datasets
    """
    # Base path to extracted data
    base_path = "/media/yugi/MAIN DRIVE/RoboFUSE_Dataset/Extracted"
    
    # List all available scenarios
    radar_data_path = os.path.join(base_path, "Extracted_Radar_Data")
    vicon_data_path = os.path.join(base_path, "vicon_data")
    
    if os.path.exists(radar_data_path) and os.path.exists(vicon_data_path):
        radar_scenarios = [d for d in os.listdir(radar_data_path) 
                        if os.path.isdir(os.path.join(radar_data_path, d))]
        vicon_scenarios = [d for d in os.listdir(vicon_data_path) 
                        if os.path.isdir(os.path.join(vicon_data_path, d))]
        
        # Find common scenarios
        common_scenarios = sorted(list(set(radar_scenarios) & set(vicon_scenarios)))
        
        if not process_all:
            print("\nAvailable scenarios:")
            for i, scenario in enumerate(common_scenarios):
                print(f"  [{i}] {scenario}")
            
            # Default to CPPS_Horizontal if available
            default_scenario = "CPPS_Horizontal"
            if default_scenario in common_scenarios:
                scenario_idx = common_scenarios.index(default_scenario)
            else:
                scenario_idx = 0
            
            # Let user choose scenario
            try:
                choice = input(f"\nEnter scenario number (default: {scenario_idx} - {common_scenarios[scenario_idx]}): ")
                if choice.strip():
                    scenario_idx = int(choice)
                scenarios_to_process = [common_scenarios[scenario_idx]]
            except (ValueError, IndexError):
                print(f"Invalid choice, using default scenario: {common_scenarios[scenario_idx]}")
                scenarios_to_process = [common_scenarios[scenario_idx]]
        else:
            # Process all scenarios
            scenarios_to_process = common_scenarios
            print(f"\nProcessing all {len(scenarios_to_process)} scenarios automatically")
    else:
        print("Warning: Could not find radar or vicon data paths. Using default scenario.")
        scenarios_to_process = ["CPPS_Horizontal"]
    
    # Process each selected scenario
    for scenario in scenarios_to_process:
        print(f"\n{'='*50}")
        print(f"Processing scenario: {scenario}")
        print(f"{'='*50}")
        
        # Initialize preprocessor
        preprocessor = RoboFUSEPreprocessor(base_path, scenario)
        
        if not process_all:
            # Print available data
            preprocessor.print_available_data()
        
        # Get available files
        robot1_dirs = preprocessor.find_matching_files("Robot_1")
        robot2_dirs = preprocessor.find_matching_files("Robot_2")
        vicon_files = preprocessor.find_vicon_files()
        
        if not robot1_dirs or not robot2_dirs or not vicon_files:
            print(f"Error: Missing data for processing scenario {scenario}, skipping")
            continue
        
        # Try to match datasets based on timestamps in filenames
        matched_sets = []
        
        # Helper function to extract timestamp from filenames
        def extract_timestamp(filename):
            # Extract date and time parts (like "20250219_120022")
            parts = filename.split('_')
            if len(parts) >= 2:
                try:
                    date = parts[0]
                    time = parts[1]
                    # Handle case where time might contain non-numeric parts
                    time = ''.join([c for c in time if c.isdigit()])[:6]  # Keep first 6 digits
                    if date.isdigit() and time.isdigit():
                        return date + time  # Return combined timestamp like "20250219120022"
                except:
                    pass
            return filename  # Return original if parsing fails
        
        # Convert all filenames to timestamps for comparison
        r1_timestamps = {r1_dir: extract_timestamp(r1_dir) for r1_dir in robot1_dirs}
        r2_timestamps = {r2_dir: extract_timestamp(r2_dir) for r2_dir in robot2_dirs}
        
        # For Vicon files, extract timestamp from middle of filename (like "CPPS_vertical_20250219_120120.csv")
        vicon_timestamps = {}
        for v_file in vicon_files:
            parts = v_file.split('_')
            if len(parts) >= 3:
                try:
                    date = parts[-2]  # Second to last part should be date
                    time = parts[-1].split('.')[0]  # Last part before extension should be time
                    if date.isdigit() and time.isdigit():
                        vicon_timestamps[v_file] = date + time
                    else:
                        vicon_timestamps[v_file] = extract_timestamp(v_file)
                except:
                    vicon_timestamps[v_file] = extract_timestamp(v_file)
            else:
                vicon_timestamps[v_file] = extract_timestamp(v_file)
        
        # For each Robot_1 directory, find best matching Robot_2 and Vicon
        for r1_dir in robot1_dirs:
            r1_date = r1_dir.split('_')[0]  # Get date part for basic filtering
            r1_ts = r1_timestamps[r1_dir]
            
            # Find matching robot 2 directories by date
            matching_r2_dirs = [r2 for r2 in robot2_dirs if r2.startswith(r1_date)]
            
            if matching_r2_dirs:
                # Find closest Robot_2 directory by timestamp
                if len(matching_r2_dirs) == 1:
                    best_r2 = matching_r2_dirs[0]
                else:
                    # Find the closest timestamp
                    best_r2 = min(matching_r2_dirs, 
                                 key=lambda r2: abs(int(r1_ts) - int(r2_timestamps[r2])) 
                                 if r1_ts.isdigit() and r2_timestamps[r2].isdigit() else float('inf'))
                
                # Find matching vicon files by date
                matching_vicon = [v for v in vicon_files if r1_date in v]
                
                if matching_vicon:
                    # Find closest Vicon file by timestamp
                    if len(matching_vicon) == 1:
                        best_vicon = matching_vicon[0]
                    else:
                        # Find the closest timestamp
                        best_vicon = min(matching_vicon, 
                                        key=lambda v: abs(int(r1_ts) - int(vicon_timestamps[v]))
                                        if r1_ts.isdigit() and vicon_timestamps[v].isdigit() else float('inf'))
                    
                    # Add this matched set
                    matched_sets.append({
                        'robot1_dir': r1_dir,
                        'robot2_dir': best_r2,
                        'vicon_file': best_vicon,
                        # Calculate a match score (lower is better)
                        'score': (abs(int(r1_ts) - int(r2_timestamps[best_r2])) +
                                 abs(int(r1_ts) - int(vicon_timestamps[best_vicon])))
                        if r1_ts.isdigit() and r2_timestamps[best_r2].isdigit() and vicon_timestamps[best_vicon].isdigit()
                        else float('inf')
                    })
        
        # Sort matched sets by score (best matches first)
        matched_sets.sort(key=lambda x: x.get('score', float('inf')))
        
        if not matched_sets:
            print(f"No matching sets found for scenario {scenario}, skipping")
            continue
            
        if not process_all:
            print("\nFound matching data sets:")
            for i, match in enumerate(matched_sets):
                print(f"  [{i}] Robot1: {match['robot1_dir']}, Robot2: {match['robot2_dir']}, Vicon: {match['vicon_file']}")
            
            # Let user choose a set
            try:
                choice = input("\nEnter set number to process (default: 0): ")
                if choice.strip():
                    set_idx = int(choice)
                else:
                    set_idx = 0
                sets_to_process = [matched_sets[set_idx]]
            except (ValueError, IndexError):
                print(f"Invalid choice, using first matched set")
                sets_to_process = [matched_sets[0]]
        else:
            # Process all matched sets
            sets_to_process = matched_sets
            print(f"Found {len(sets_to_process)} matching data sets to process")
        
        # Process each selected set
        for set_idx, selected_set in enumerate(sets_to_process):
            robot1_dir = selected_set['robot1_dir']
            robot2_dir = selected_set['robot2_dir']
            vicon_file = selected_set['vicon_file']
            
            print(f"\nProcessing data set {set_idx+1}/{len(sets_to_process)}:")
            print(f"  Robot1: {robot1_dir}")
            print(f"  Robot2: {robot2_dir}")
            print(f"  Vicon: {vicon_file}")
            
            try:
                processed_data = preprocessor.process_data(
                    robot1_dir=robot1_dir,
                    robot2_dir=robot2_dir,
                    vicon_file=vicon_file,
                    resample_interval='25ms',
                    save_results=True
                )
                
                if processed_data and processed_data['final_synchronized_data'] is not None:
                    # Print summary of synchronized data
                    final_df = processed_data['final_synchronized_data']
                    
                    print("\n=== Synchronized Data Summary ===")
                    print(f"Number of samples: {len(final_df)}")
                    print(f"Columns: {final_df.columns.tolist()}")
                    
                    # Print sample data (first 5 rows) only in interactive mode
                    if not process_all:
                        print("\nSample data (first 5 rows):")
                        pd.set_option('display.max_columns', 10)
                        print(final_df.head())
                        
                        # Plot synchronized data in interactive mode
                        preprocessor.plot_synchronized_data()
                    
                    print(f"\nPreprocessing completed successfully for set {set_idx+1}!")
                    
                    # Print output file paths
                    output_path = os.path.join(preprocessor.output_dir, preprocessor.scenario)
                    print(f"\nOutput files saved to: {output_path}")
                else:
                    print(f"Error: Processing failed for set {set_idx+1}")
            except Exception as e:
                print(f"Error processing data set {set_idx+1}: {str(e)}")
                continue
    
    print("\nAll processing completed!")


if __name__ == "__main__":
    import sys
    # Check if "--all" flag is provided
    process_all = "--all" in sys.argv
    main(process_all)


if __name__ == "__main__":
    main()
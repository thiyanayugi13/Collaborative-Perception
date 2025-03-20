#!/usr/bin/env python3

import argparse
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess
import json
import logging
from datetime import datetime


class CollaborativePerceptionPipeline:
    """
    An integrated pipeline for the collaborative perception framework.
    
    This script ties together data preprocessing, coordinate transformation,
    global map visualization, and accuracy evaluation into a single workflow.
    """
    
    def __init__(self, config=None):
        """
        Initialize the pipeline with configuration parameters.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config or {}
        
        # Set up logging
        self.setup_logging()
        
        # Default configuration parameters
        self.default_config = {
            'input_dir': None,
            'output_dir': 'output',
            'scenario': 'CPPS_Horizontal',
            'robot1_dir': None,
            'robot2_dir': None,
            'vicon_file': None,
            'generate_plots': True,
            'save_plots': True,
            'plot_formats': ['png'],
            'evaluation_metrics': ['rmse', 'iou', 'point_cloud']
        }
        
        # Update configuration with provided values
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
        
        # Paths to component scripts
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.script_paths = {
            'coordinate_transformation': os.path.join(script_dir, 'coordinate_transformation.py'),
            'visualize_global_map': os.path.join(script_dir, 'visualize_global_map.py'),
            'evaluate_accuracy': os.path.join(script_dir, 'evaluate_accuracy.py')
        }
        
        # Verify script paths
        self.verify_script_paths()
        
        # Create output directory
        self.create_output_directory()
        
    def setup_logging(self):
        """Set up logging configuration"""
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"cp_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            ]
        )
        self.logger = logging.getLogger("CollaborativePerceptionPipeline")
    
    def verify_script_paths(self):
        """Verify that all component scripts exist"""
        missing_scripts = []
        for name, path in self.script_paths.items():
            if not os.path.exists(path):
                missing_scripts.append(f"{name}: {path}")
        
        if missing_scripts:
            self.logger.error(f"Missing component scripts: {', '.join(missing_scripts)}")
            self.logger.error("Please ensure all component scripts are in the same directory as this pipeline script.")
            raise FileNotFoundError(f"Missing component scripts: {', '.join(missing_scripts)}")
    
    def create_output_directory(self):
        """Create the output directory structure"""
        if self.config['output_dir']:
            os.makedirs(self.config['output_dir'], exist_ok=True)
            
            # Create subdirectories
            subdirs = ['transformed_data', 'visualizations', 'evaluation_results']
            for subdir in subdirs:
                os.makedirs(os.path.join(self.config['output_dir'], subdir), exist_ok=True)
            
            self.logger.info(f"Created output directory structure at: {self.config['output_dir']}")
    
    def find_input_files(self):
        """
        Find input files for processing if not explicitly specified.
        
        Returns:
            bool: True if files were found, False otherwise
        """
        if not self.config['input_dir']:
            self.logger.error("Input directory not specified")
            return False
        
        sync_data_dir = os.path.join(self.config['input_dir'], self.config['scenario'])
        if not os.path.exists(sync_data_dir):
            self.logger.error(f"Scenario directory not found: {sync_data_dir}")
            return False
        
        # List available synchronized datasets
        available_datasets = []
        for item in os.listdir(sync_data_dir):
            item_path = os.path.join(sync_data_dir, item)
            if os.path.isdir(item_path):
                # Check if it contains the necessary files
                final_sync_files = [f for f in os.listdir(item_path) if f.startswith('final_synchronized_')]
                if final_sync_files:
                    available_datasets.append({
                        'dir_name': item,
                        'path': item_path,
                        'sync_file': os.path.join(item_path, final_sync_files[0])
                    })
        
        if not available_datasets:
            self.logger.error(f"No synchronized datasets found in {sync_data_dir}")
            return False
        
        # If specific dirs not provided, use the first available dataset
        if not (self.config['robot1_dir'] and self.config['robot2_dir'] and self.config['vicon_file']):
            self.logger.info(f"Using first available dataset: {available_datasets[0]['dir_name']}")
            self.input_file = available_datasets[0]['sync_file']
            
            # Extract component parts from the directory name
            parts = available_datasets[0]['dir_name'].split('_')
            if len(parts) >= 2:
                # Simplified parsing - actual implementation might need more robust handling
                self.config['robot1_dir'] = parts[0]
                self.config['robot2_dir'] = parts[1]
                self.config['vicon_file'] = '_'.join(parts[2:])
        else:
            # Find the specific dataset
            dataset_name = f"{self.config['robot1_dir']}_{self.config['robot2_dir']}_{self.config['vicon_file']}"
            matching_datasets = [d for d in available_datasets if d['dir_name'] == dataset_name]
            
            if not matching_datasets:
                self.logger.error(f"Specified dataset not found: {dataset_name}")
                return False
                
            self.input_file = matching_datasets[0]['sync_file']
        
        self.logger.info(f"Selected input file: {self.input_file}")
        return True
    
    def run_coordinate_transformation(self):
        """
        Run the coordinate transformation component.
        
        Returns:
            str: Path to the transformed data file
        """
        self.logger.info("Running coordinate transformation...")
        
        # Output file path
        output_file = os.path.join(
            self.config['output_dir'], 
            'transformed_data', 
            f"transformed_{os.path.basename(self.input_file)}"
        )
        
        # Construct command
        cmd = [
            'python3', self.script_paths['coordinate_transformation'],
            '--input', self.input_file,
            '--output', output_file
        ]
        
        # Only add visualization if requested
        if self.config['generate_plots']:
            plot_type = '2d'  # Default to 2D plot
            cmd.extend(['--plot', plot_type])
            
            if self.config['save_plots']:
                plot_file = os.path.join(
                    self.config['output_dir'], 
                    'visualizations', 
                    f"transform_{os.path.basename(self.input_file).replace('.csv', '.png')}"
                )
                cmd.extend(['--save-plot', plot_file])
        
        # Run the command
        self.logger.info(f"Running command: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.logger.info(result.stdout)
            if result.stderr:
                self.logger.warning(result.stderr)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error running coordinate transformation: {e}")
            self.logger.error(e.stderr)
            return None
        
        if not os.path.exists(output_file):
            self.logger.error(f"Transformation failed: output file not created at {output_file}")
            return None
            
        self.logger.info(f"Coordinate transformation complete: {output_file}")
        return output_file
    
    def run_visualization(self, transformed_file):
        """
        Run the visualization component.
        
        Args:
            transformed_file (str): Path to the transformed data file
            
        Returns:
            bool: True if visualization was successful, False otherwise
        """
        if not transformed_file or not os.path.exists(transformed_file):
            self.logger.error(f"Invalid transformed file for visualization: {transformed_file}")
            return False
            
        self.logger.info("Running global map visualization...")
        
        # Generate different visualization types
        visualization_types = ['2d', '3d', 'density']
        success = True
        
        for viz_type in visualization_types:
            # Output file path
            output_file = os.path.join(
                self.config['output_dir'], 
                'visualizations', 
                f"{viz_type}_map_{os.path.basename(transformed_file).replace('.csv', '.png')}"
            )
            
            # Construct command
            cmd = [
                'python3', self.script_paths['visualize_global_map'],
                '--input', transformed_file,
                '--plot-type', viz_type,
                '--output', output_file,
                '--analyze'
            ]
            
            # Run the command
            self.logger.info(f"Running visualization command ({viz_type}): {' '.join(cmd)}")
            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                self.logger.info(result.stdout)
                if result.stderr:
                    self.logger.warning(result.stderr)
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Error running visualization ({viz_type}): {e}")
                self.logger.error(e.stderr)
                success = False
                continue
            
            if not os.path.exists(output_file):
                self.logger.warning(f"Visualization output file not created: {output_file}")
                success = False
            else:
                self.logger.info(f"Generated {viz_type} visualization: {output_file}")
        
        return success
    
    def run_evaluation(self, transformed_file):
        """
        Run the evaluation component.
        
        Args:
            transformed_file (str): Path to the transformed data file
            
        Returns:
            dict: Evaluation results
        """
        if not transformed_file or not os.path.exists(transformed_file):
            self.logger.error(f"Invalid transformed file for evaluation: {transformed_file}")
            return None
            
        self.logger.info("Running transformation accuracy evaluation...")
        
        # Output file path
        output_file = os.path.join(
            self.config['output_dir'], 
            'evaluation_results', 
            f"evaluation_{os.path.basename(transformed_file).replace('.csv', '.json')}"
        )
        
        # Construct command
        cmd = [
            'python3', self.script_paths['evaluate_accuracy'],
            '--input', transformed_file,
            '--output', output_file
        ]
        
        # Add plot options if requested
        if self.config['generate_plots'] and self.config['save_plots']:
            plot_dir = os.path.join(
                self.config['output_dir'], 
                'visualizations', 
                f"evaluation_{os.path.basename(transformed_file).replace('.csv', '')}"
            )
            cmd.extend(['--save-plots', plot_dir])
        
        # Run the command
        self.logger.info(f"Running evaluation command: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.logger.info(result.stdout)
            if result.stderr:
                self.logger.warning(result.stderr)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error running evaluation: {e}")
            self.logger.error(e.stderr)
            return None
        
        if not os.path.exists(output_file):
            self.logger.error(f"Evaluation output file not created: {output_file}")
            return None
            
        # Load and return evaluation results
        try:
            with open(output_file, 'r') as f:
                evaluation_results = json.load(f)
                
            self.logger.info(f"Evaluation complete: {output_file}")
            return evaluation_results
        except Exception as e:
            self.logger.error(f"Error loading evaluation results: {e}")
            return None
    
    def run_pipeline(self):
        """
        Run the complete pipeline.
        
        Returns:
            dict: Pipeline results summary
        """
        start_time = time.time()
        self.logger.info(f"Starting Collaborative Perception Pipeline for scenario: {self.config['scenario']}")
        
        # Find input files
        if not self.find_input_files():
            self.logger.error("Failed to find input files")
            return {'status': 'error', 'message': 'Failed to find input files'}
        
        # Run coordinate transformation
        transformed_file = self.run_coordinate_transformation()
        if not transformed_file:
            self.logger.error("Coordinate transformation failed")
            return {'status': 'error', 'message': 'Coordinate transformation failed'}
        
        # Run visualization
        visualization_success = self.run_visualization(transformed_file)
        if not visualization_success:
            self.logger.warning("Some visualization steps failed")
        
        # Run evaluation
        evaluation_results = self.run_evaluation(transformed_file)
        if not evaluation_results:
            self.logger.warning("Evaluation failed")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Compile results summary
        results_summary = {
            'status': 'success',
            'scenario': self.config['scenario'],
            'input_file': self.input_file,
            'transformed_file': transformed_file,
            'processing_time_seconds': processing_time,
            'evaluation_results': evaluation_results
        }
        
        self.logger.info(f"Pipeline completed in {processing_time:.2f} seconds")
        return results_summary
    
    def save_summary(self, results_summary):
        """
        Save the results summary to a file.
        
        Args:
            results_summary (dict): Pipeline results summary
            
        Returns:
            str: Path to the summary file
        """
        if not results_summary:
            self.logger.error("No results summary to save")
            return None
            
        # Create summary file path
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = os.path.join(
            self.config['output_dir'], 
            f"pipeline_summary_{timestamp}.json"
        )
        
        # Save to file
        try:
            with open(summary_file, 'w') as f:
                json.dump(results_summary, f, indent=2)
                
            self.logger.info(f"Saved pipeline summary to: {summary_file}")
            return summary_file
        except Exception as e:
            self.logger.error(f"Error saving summary: {e}")
            return None


def main():
    """Main function to run the collaborative perception pipeline."""
    parser = argparse.ArgumentParser(description='Run the collaborative perception pipeline')
    parser.add_argument('--input-dir', '-i', type=str, required=True,
                        help='Path to the synchronized data directory')
    parser.add_argument('--output-dir', '-o', type=str, default='output',
                        help='Path to store output files (default: output)')
    parser.add_argument('--scenario', '-s', type=str, default='CPPS_Horizontal',
                        help='Scenario to process (default: CPPS_Horizontal)')
    parser.add_argument('--robot1-dir', type=str, default=None,
                        help='Directory containing Robot 1 data (default: auto-detect)')
    parser.add_argument('--robot2-dir', type=str, default=None,
                        help='Directory containing Robot 2 data (default: auto-detect)')
    parser.add_argument('--vicon-file', type=str, default=None,
                        help='Vicon data file (default: auto-detect)')
    parser.add_argument('--no-plots', action='store_true',
                        help='Disable plot generation')
    parser.add_argument('--no-save-plots', action='store_true',
                        help='Do not save generated plots')
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        'input_dir': args.input_dir,
        'output_dir': args.output_dir,
        'scenario': args.scenario,
        'robot1_dir': args.robot1_dir,
        'robot2_dir': args.robot2_dir,
        'vicon_file': args.vicon_file,
        'generate_plots': not args.no_plots,
        'save_plots': not args.no_save_plots
    }
    
    # Create and run pipeline
    pipeline = CollaborativePerceptionPipeline(config)
    results = pipeline.run_pipeline()
    
    # Save summary
    pipeline.save_summary(results)
    
    return 0


if __name__ == "__main__":
    main()
#!/bin/bash

# Base directories
INPUT_DIR="/media/yugi/MAIN DRIVE/RoboFUSE_Dataset/SynchronizedData"
OUTPUT_DIR="output/all_datasets_result"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Process all scenarios and their datasets
for scenario_dir in "$INPUT_DIR"/*; do
  if [ -d "$scenario_dir" ]; then
    scenario=$(basename "$scenario_dir")
    echo "Processing scenario: $scenario"
    
    # Create scenario-specific output directory
    mkdir -p "$OUTPUT_DIR/$scenario"
    
    # Process each dataset within this scenario
    for dataset_dir in "$scenario_dir"/*; do
      if [ -d "$dataset_dir" ]; then
        dataset=$(basename "$dataset_dir")
        echo "  Processing dataset: $dataset"
        
        # Extract robot1_dir, robot2_dir, and vicon_file from the dataset name
        # Format is typically: [robot1_dir]_[robot2_dir]_[vicon_file]
        IFS='_' read -ra PARTS <<< "$dataset"
        
        # First part is robot1_dir
        robot1_dir="${PARTS[0]}"
        
        # Second part is robot2_dir
        robot2_dir="${PARTS[1]}"
        
        # The rest is the vicon_file
        vicon_file="${dataset#${robot1_dir}_${robot2_dir}_}"
        
        # Create dataset-specific output directory
        dataset_output_dir="$OUTPUT_DIR/$scenario/$dataset"
        mkdir -p "$dataset_output_dir"
        
        # Find the synchronized data file
        sync_file=$(find "$dataset_dir" -name "final_synchronized_*.csv" | head -n 1)
        
        if [ -n "$sync_file" ]; then
          echo "    Found synchronized data: $(basename "$sync_file")"
          
          # Run the transformation directly on this file
          output_file="$dataset_output_dir/transformed_$(basename "$sync_file")"
          
          echo "    Running coordinate transformation..."
          python3 coordinate_transformation.py \
            --input "$sync_file" \
            --output "$output_file" \
            --plot 2d \
            --save-plot "$dataset_output_dir/transform_visualization.png"
            
          # Now run visualization
          echo "    Generating visualizations..."
          for viz_type in 2d 3d density; do
            python3 visualize_global_map.py \
              --input "$output_file" \
              --plot-type "$viz_type" \
              --output "$dataset_output_dir/${viz_type}_map.png" \
              --analyze
          done
          
          # Finally run evaluation
          echo "    Evaluating transformation accuracy..."
          python3 evaluate_accuracy.py \
            --input "$output_file" \
            --output "$dataset_output_dir/evaluation_results.json" \
            --save-plots "$dataset_output_dir/evaluation_plots"
            
          echo "    Dataset processing complete"
        else
          echo "    ERROR: No synchronized data file found in $dataset_dir"
        fi
        
        echo "  ----------------------------------"
      fi
    done
    
    echo "Scenario $scenario completed"
    echo "========================================"
  fi
done

echo "All scenarios and datasets processed successfully!"
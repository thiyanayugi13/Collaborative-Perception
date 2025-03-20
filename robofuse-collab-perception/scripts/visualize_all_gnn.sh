#!/bin/bash

# Base directory containing GNN data
BASE_DIR="/media/yugi/MAIN DRIVE/RoboFUSE_Dataset/GNN_Ready_data"

# Filtering parameters
MIN_OCCUPANCY=0.2
MIN_HEIGHT=0.05
MAX_HEIGHT=2.0

# Find all .pt files in the directory structure
find "$BASE_DIR" -name "*_gnn.pt" | while read -r file; do
    echo "Processing: $file"
    python3 visualize_gnn.py --input "$file" \
                             --min-occupancy $MIN_OCCUPANCY \
                             --min-height $MIN_HEIGHT \
                             --max-height $MAX_HEIGHT
    echo "Completed: $file"
    echo "------------------------"
done

echo "All files processed!"
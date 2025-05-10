#! /bin/bash

# Get isaaclab directory from command line argument, default to current directory
ISAACLAB_DIR=${1:-"./"}

# Check if directory exists
if [ ! -d "$ISAACLAB_DIR" ]; then
    echo "Error: Directory $ISAACLAB_DIR does not exist"
    exit 1
fi

# Get all folders in current directory except .git
folders=$(find . -maxdepth 1 -type d ! -name ".*" ! -name "$(basename $(pwd))")

# Create symbolic links for each folder
for folder in $folders; do
    folder_name=$(basename "$folder")

    if [[ "$folder_name" == "algorithms" ]]; then
        continue
    fi

    target_dir="$ISAACLAB_DIR/source/isaaclab_tasks/isaaclab_tasks/direct/$folder_name"
    
    # Create symbolic link if it doesn't already exist
    if [ ! -L "$target_dir" ]; then
        ln -s "$(pwd)/$folder_name" "$target_dir"
        echo "Created symbolic link for $folder_name"
    else
        echo "Symbolic link for $folder_name already exists"
    fi
done

echo "Finished creating symbolic links"


import os
import shutil

def move_files(root_dir, move_dir, extensions):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            # Check if the file has one of the desired extensions
            if any(filename.endswith(ext) for ext in extensions):
                source_path = os.path.join(dirpath, filename)
                
                # Construct the target directory and file paths
                relative_path = os.path.relpath(dirpath, root_dir)
                target_dir = os.path.join(move_dir, relative_path)
                target_path = os.path.join(target_dir, filename)
                
                # Create the target directory if it doesn't exist
                os.makedirs(target_dir, exist_ok=True)
                
                # Move the file only if it doesn't already exist in the target directory
                if not os.path.exists(target_path):
                    shutil.move(source_path, target_path)
                    print(f"Moved {source_path} to {target_path}")
                else:
                    print(f"Target file {target_path} already exists. Skipping.")

# Define the root directory where the original files are located
move_dir = '/Users/jeonsang-eon/sleep_data/'
# Define the destination directory where the files should be moved
root_dir = '/Users/jeonsang-eon/sleep_data_processed/'
# List of file extensions to move
extensions = ['.vmrk', '.eeg']

# Execute the file moving
move_files(root_dir, move_dir, extensions)

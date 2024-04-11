import os
import shutil

def copy_files_if_not_exist(src_dir, dest_dir, extensions):
    """
    Copy files with specific extensions from src_dir to dest_dir, maintaining the directory structure.
    Only copies files if they do not already exist in the destination directory.
    
    :param src_dir: Source directory
    :param dest_dir: Destination directory
    :param extensions: List of file extensions to copy
    """
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith(tuple(extensions)):
                src_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(src_file_path, src_dir)
                dest_file_path = os.path.join(dest_dir, relative_path)
                
                # Check if the destination file exists
                if not os.path.exists(dest_file_path):
                    # Ensure the destination directory exists
                    os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)
                    # Copy the file
                    shutil.copy(src_file_path, dest_file_path)
                    print(f"Copied {src_file_path} to {dest_file_path}")
                else:
                    print(f"File already exists, skipping: {dest_file_path}")

# Define your source and destination directories
src_directory = '/Users/jeonsang-eon/sleep_data/'
dest_directory = '/Users/jeonsang-eon/sleep_data_processed/'

# List of extensions to copy
extensions_to_copy = ['.vmrk', '.eeg']

# Copy the files
copy_files_if_not_exist(src_directory, dest_directory, extensions_to_copy)

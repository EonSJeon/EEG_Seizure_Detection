import os

def delete_files_with_ref(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if '_ref' in filename:
                # Construct the full path to the file
                file_path = os.path.join(dirpath, filename)
                # Delete the file
                os.remove(file_path)
                print(f"Deleted: {file_path}")

# Define the root directory to search
root_dir = '/Users/jeonsang-eon/sleep_data/'

# Call the function
delete_files_with_ref(root_dir)

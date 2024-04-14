import os
import re

def rename_files(root_dir):
    """
    Rename files in the specified directory and its subdirectories
    from '#s.extension' to '#.extension', where # is a number.
    """
    # Regex to identify and capture the required filename pattern
    pattern = re.compile(r'(\d+)s(\.\w+)$')  # Matches '123s.npy', capturing '123' and '.npy'

    # Walk through all directories and subdirectories
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            # Check if the filename matches the pattern
            match = pattern.search(filename)
            if match:
                # Construct the new filename
                new_filename = f"{match.group(1)}{match.group(2)}"
                # Get the full path of the current file and the new file
                old_filepath = os.path.join(dirpath, filename)
                new_filepath = os.path.join(dirpath, new_filename)

                # Rename the file
                os.rename(old_filepath, new_filepath)
                print(f"Renamed '{old_filepath}' to '{new_filepath}'")

# Specify the root directory
root_dir = '/Users/jeonsang-eon/sleep_data_processed'
rename_files(root_dir)

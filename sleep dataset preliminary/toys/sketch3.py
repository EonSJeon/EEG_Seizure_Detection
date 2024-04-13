# import os

# def print_file_hierarchy(root_dir, indent=0):
#     """Recursively prints the file hierarchy of a given directory with indentation representing the hierarchy.

#     Args:
#         root_dir (str): The root directory from which to start listing the hierarchy.
#         indent (int): The indentation level (used in recursive calls to increase the indentation).
#     """
#     # Get all the entries in the directory sorted by name
#     try:
#         entries = sorted(os.listdir(root_dir))
#     except PermissionError:
#         print(" " * indent + "PermissionError: Cannot access contents of", root_dir)
#         return

#     for entry in entries:
#         full_path = os.path.join(root_dir, entry)
#         if os.path.isdir(full_path):
#             print(" " * indent + f"[Folder] {entry}")
#             # Recurse into the subdirectory
#             print_file_hierarchy(full_path, indent + 4)
#         else:
#             print(" " * indent + f"{entry}")

# # Define the root directory you want to inspect
# root_dir = '/Users/jeonsang-eon/sleep_data_processed/'

# # Call the function
# print_file_hierarchy(root_dir)
import os
from collections import defaultdict

def count_files_by_format(root_dir):
    """Counts and prints the number of files per format for each subfolder in the given directory.

    Args:
        root_dir (str): The root directory from which to start counting files by their format.
    """
    # Dictionary to hold file counts per format for each subfolder
    format_counts = defaultdict(lambda: defaultdict(int))

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Process each file in the directory
        for filename in filenames:
            # Extract file extension
            _, ext = os.path.splitext(filename)
            ext = ext.lower()  # Normalize the extension to lowercase to avoid duplicates like .JPG and .jpg
            if ext:  # Ensure there is an extension
                format_counts[dirpath][ext] += 1

    # Print the results
    for subdir, counts in format_counts.items():
        print(f"Folder: {subdir}")
        for ext, count in counts.items():
            print(f"  {ext}: {count}")
        print()  # Add a newline for better separation between folders

# Define the root directory you want to inspect
root_dir = '/Users/jeonsang-eon/sleep_data_processed/'

# Call the function
count_files_by_format(root_dir)

expected_folders = {f"sub-{i:02d}" for i in range(1, 34)}  # Generate a set from sub-01 to sub-33
actual_folders = {
    "sub-13", "sub-14", "sub-22", "sub-25", "sub-24", "sub-23", "sub-15", "sub-12", "sub-30", 
    "sub-08", "sub-01", "sub-06", "sub-07", "sub-09", "sub-31", "sub-17", "sub-28", "sub-10",
    "sub-26", "sub-19", "sub-21", "sub-20", "sub-27", "sub-18", "sub-11", "sub-16", "sub-29",
    "sub-33", "sub-05", "sub-02", "sub-03", "sub-04"
}

# Determine missing folders
missing_folders = expected_folders - actual_folders

# Print the results
print(f"Total folders listed: {len(actual_folders)}")
print(f"Missing folders: {sorted(missing_folders)}")

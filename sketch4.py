import os
import shutil

root_dir = '/Users/jeonsang-eon/ds003768-download'  # Root directory containing the 'sub-xx' directories

# Iterate through all 'sub-xx' directories
for i in range(1, 34):  # Assuming 'xx' goes from 01 to 33
    sub_dir = os.path.join(root_dir, f'sub-{i:02}')  # Format sub-directory name

    # Check and delete 'anat' directory if it exists
    anat_dir = os.path.join(sub_dir, 'anat')
    if os.path.exists(anat_dir) and os.path.isdir(anat_dir):
        shutil.rmtree(anat_dir)
        print(f"Deleted '{anat_dir}'")

    # Move files from 'eeg' to 'sub-xx' and delete 'eeg'
    eeg_dir = os.path.join(sub_dir, 'eeg')
    if os.path.exists(eeg_dir) and os.path.isdir(eeg_dir):
        for filename in os.listdir(eeg_dir):
            source_file = os.path.join(eeg_dir, filename)
            destination_file = os.path.join(sub_dir, filename)

            # Move each file
            shutil.move(source_file, destination_file)
            print(f"Moved '{source_file}' to '{destination_file}'")

        # Delete the now-empty 'eeg' directory
        os.rmdir(eeg_dir)
        print(f"Deleted empty directory '{eeg_dir}'")


import os
import subprocess

def check_file_format(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    for line in lines:
        if line.count('=') != 1:
            # Found a line with formatting issue
            return True
    return False

# List to hold directories with problematic band.conf files
problematic_dirs = []

# Walk through all directories in the current directory
for root, dirs, files in os.walk('.'):
    for dir in dirs:
        band_conf_path = os.path.join(root, dir, 'band.conf')
        if os.path.isfile(band_conf_path):
            if check_file_format(band_conf_path):
                # Add the directory to the list if a problem is found
                problematic_dirs.append(os.path.join(root, dir))

if problematic_dirs:
    print("Found problematic band.conf files in the following directories:")
    for dir in problematic_dirs:
        print(dir)
else:
    print("No formatting issues found in band.conf files.")
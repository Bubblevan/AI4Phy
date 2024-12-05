import os
import subprocess

# Function to add a line to band.conf if it doesn't already exist
def add_line_to_file(file_path, line_to_add):
    with open(file_path, 'r') as file:
        contents = file.readlines()
    
    # 去除每行末尾的空白字符后进行检查
    clean_contents = [line.rstrip() for line in contents]
    if line_to_add not in clean_contents:
        with open(file_path, 'a') as file:  # 使用追加模式打开文件
            file.write(line_to_add + "\\n")  # 确保添加换行符
        print(f"Added line to {file_path}")
    else:
        print(f"Line already exists in {file_path}")

# Walk through all directories in the current directory
for root, dirs, files in os.walk('.'):
    for dir in dirs:
        # Construct the path to the potential band.conf file
        band_conf_path = os.path.join(root, dir, 'band.conf')
        # Check if band.conf exists in this directory
        if os.path.isfile(band_conf_path):
            # Add the required line to band.conf
            add_line_to_file(band_conf_path, 'FULL_FORCE_CONSTANTS = .TRUE.')
            # Change to the directory containing band.conf
            os.chdir(os.path.join(root, dir))
            # Execute the Phonopy command
            subprocess.run(['phonopy', '-p', 'band.conf'])
            # Change back to the original directory to continue the walk
            os.chdir(root)
import os
import random
from graph import CrystalGraphDataset , CrystalDataset
from torch_geometric.loader import DataLoader
import torch
import concurrent.futures

directory='./data'
datasets = []
# 获取目录下所有文件
files_and_dirs = os.listdir(directory)
dirs = [f for f in files_and_dirs if os.path.isdir(os.path.join(directory, f))]

def generate_loader_cpu(dirs , directory = './data', loader_size=80,batch_size=8):
    """
    generate a batch size 80
    cpu
    """
    # choose 80 files randomly or all if less than loader_size
    batch_dirs = random.sample(dirs, min(loader_size, len(dirs)))
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_dataset, dir_name) for dir_name in batch_dirs]
        datasets = [future.result() for future in concurrent.futures.as_completed(futures)]
    data_loader = DataLoader(CrystalDataset(datasets), batch_size = batch_size, shuffle=True)
    # remove the batch files
    dirs = [f for f in dirs if f not in batch_dirs]
    return data_loader , dirs

def generate_loader_cuda(dirs , directory = './data', loader_size=80,batch_size=8):
    """
    generate a batch size 80
    cuda
    """

    batch_dirs = random.sample(dirs, min(loader_size, len(dirs)))
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_dataset, dir_name) for dir_name in batch_dirs]
        datasets = [future.result().to('cuda') for future in concurrent.futures.as_completed(futures)]
    data_loader = DataLoader(CrystalDataset(datasets), batch_size=batch_size, shuffle=True,generator=torch.Generator(device='cuda'))
    dirs = [f for f in dirs if f not in batch_dirs]
    return data_loader , dirs

def process_dataset(dir_name):
    force_constants_file_path = os.path.join(directory, dir_name, "FORCE_CONSTANTS")
    phonopy_file_path = os.path.join(directory, dir_name, "phonopy.yaml")
    dataset = CrystalGraphDataset(phonopy_file_path, force_constants_file_path).graph_data
    return dataset
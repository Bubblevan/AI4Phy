import numpy as np
import os
import os.path as osp
import torch
from collections import namedtuple
from torch_geometric.data import Data
from torch_geometric.data import Data, InMemoryDataset, Dataset
import yaml

import concurrent.futures
import json
# skipatom_model = AtomVectors.load("skipatom/data/mat2vec.dim200.model")
Point = namedtuple('Point', ['symbol', 'coordinates', 'mass'])

class FC(InMemoryDataset):

    def __init__(self, root, split, fold_data, fold_id, fixed_size_split=True):
        assert split in ["train", "valid", "test"]
    
        self.split = split
        self.fold_data = fold_data
        self.fold_id = fold_id
        self.root = osp.abspath(root)
        self.fixed_size_split = fixed_size_split

        super().__init__(self.root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def calc_stats(self, target):
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        y = y[:, target]
        mean = float(torch.mean(y))
        mad = float(torch.mean(torch.abs(y - mean))) #median absolute deviation
        return mean, mad

    def mean(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].mean())

    def std(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].std())

    @property
    def processed_file_names(self) -> str:
        return str(self.split) +'_'+str(self.fold_id)+'.pt'

    def process(self):
        data_path ='./datasets'

        r_cut = 8
        max_neighbors = 32
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.get_graph, dir_name) for dir_name in self.fold_data]
            data_list = [future.result() for future in concurrent.futures.as_completed(futures)]

        torch.save(self.collate(data_list), self.processed_paths[0])

    def get_graph(self, dir_name):
        force_constants_file_path = os.path.join(dir_name, "FORCE_CONSTANTS")
        phonopy_file_path = os.path.join(dir_name, "phonopy.yaml")
        dataset = CrystalGraphDataset(phonopy_file_path, force_constants_file_path).graph_data
        return dataset

#create a class to store the data
class CrystalGraphDataset:
    def __init__(self, phonopy_file_path, force_constants_file_path):
        self.phonopy_file_path = phonopy_file_path
        self.force_constants_file_path = force_constants_file_path
        with open('../datasets/atom_embeddings.json', 'r') as json_file:
            self.atom_embeddings = json.load(json_file)
        self.parsed_points, self.supercell_lattice = self.read_phonopy_file()
        self.force_constants, self.atom_pairs = self.read_force_constants()
        self.graph_data = self.points_to_graph()

    def read_phonopy_file(self):
        with open(self.phonopy_file_path, 'r') as file:
            yaml_content = yaml.safe_load(file)
        supercell_points = yaml_content['supercell']['points']
        supercell_lattice = yaml_content['supercell']['lattice']
        parsed_points = [Point(p['symbol'], p['coordinates'], p['mass']) for p in supercell_points]
        return parsed_points, supercell_lattice

    # def read_phonopy_file(self):
    #     with open(self.phonopy_file_path, 'r') as file:
    #         yaml_content = yaml.safe_load(file)

    #     # 获取 unit_cell 信息
    #     unit_cell_points = yaml_content['unit_cell']['points']
    #     unit_cell_lattice = yaml_content['unit_cell']['lattice']

    #     # 解析点信息
    #     parsed_points = [Point(p['symbol'], p['coordinates'], p['mass']) for p in unit_cell_points]

    #     return parsed_points, unit_cell_lattice

    
    def read_force_constants(self):
        with open(self.force_constants_file_path, 'r') as file:
            lines = file.readlines()
        force_constants = {}
        atom_pairs = [] 
        line_index = 1  
        while line_index < len(lines):
            if lines[line_index].strip():
                atom_pair = tuple(map(int, lines[line_index].split()))
                atom_pairs.append(atom_pair)
                line_index += 1
                matrix = torch.zeros((3, 3))
                for i in range(3):
                    # matrix[i] = np.fromstring(lines[line_index], sep=' ')
                    matrix[i] = torch.tensor(list(map(float, lines[line_index].split())))
                    line_index += 1
                force_constants[atom_pair] = matrix
            else:
                line_index += 1
        # for key in force_constants.keys():
        #     force_constants[key] = torch.tensor(force_constants[key], dtype=torch.float)
        return force_constants, atom_pairs
    
    def points_to_graph(self):
        x = [p.symbol for p in self.parsed_points]
        x = [self.atom_embeddings[atom] for atom in x]
        x = torch.tensor(x, dtype=torch.float)
        lattice = torch.tensor(self.supercell_lattice, dtype=torch.float).view(3, 3)
        pos = torch.tensor([p.coordinates for p in self.parsed_points], dtype=torch.float).view(-1, 3)
        pos = torch.matmul(pos, lattice)
        pos.requires_grad_()
        #将 pos 之差小于8的原子对连接起来，视为存在边，将边的属性储存在y
        diff = pos[:, None, :] - pos[None, :, :]
        dist_squared = torch.sum(diff**2, dim=-1)
        dist_squared += torch.eye(dist_squared.shape[0]) * 1e-10
        dist_matrix = torch.sqrt(dist_squared)
        threshold = 8
        max_neighbors=32
        edge_src, edge_dst, edge_vec, distances = [], [], [], []
        num_atoms = pos.shape[0]
        print("当前样本的原子数为：", num_atoms)
        for i in range(num_atoms):
            # 获取当前原子与其他所有原子的距离并排序获取索引
            # 由于argsort是升序，排除了第一个索引（即自身），接着取前k个邻居
            neighbors_indices = dist_matrix[i].argsort()[1:max_neighbors+1]
            for neighbor_index in neighbors_indices:
                if dist_matrix[i, neighbor_index] > threshold:
                    continue
                # 确保不是自连接
                if neighbor_index == i:
                    continue
                # 添加边的源和目的地索引
                edge_src.append(i)
                edge_dst.append(neighbor_index)
                # 计算并添加边向量
                edge_vector = pos[i] - pos[neighbor_index]
                edge_vec.append(edge_vector)
                # 添加边的距离
                distances.append(dist_matrix[i, neighbor_index])

        force_constants_all = torch.zeros((num_atoms, num_atoms, 3, 3))
        for i in range(num_atoms):
            for j in range(num_atoms):
                if (i+1, j+1) in self.force_constants:
                    force_constants_all[i, j] = self.force_constants[(i+1, j+1)]
                else:
                    pass
        force_constants_all = force_constants_all.view(-1, 9)
        # 转换为Tensor
        edge_num = len(edge_src)
        edge_num = torch.tensor(edge_num, dtype=torch.long)
        edge_src = torch.tensor(edge_src, dtype=torch.long)
        edge_dst = torch.tensor(edge_dst, dtype=torch.long)
        edge_vec = torch.stack(edge_vec, dim=0)  # 堆叠列表中的向量
        edge_attr = torch.tensor(distances, dtype=torch.float, requires_grad=True)
        edge_index = torch.stack([edge_src + 1, edge_dst + 1], dim=0)
        force_constants = torch.stack([self.force_constants[tuple(edge_index[:,i].tolist())] for i in range(edge_num)])
        eigenvalues_tensor = torch.zeros((edge_num, 3), dtype=torch.float32)
        eigenvectors_tensor = torch.zeros((edge_num, 9), dtype=torch.float32)
        for i in range(edge_num):
            vals, vecs = torch.linalg.eig(force_constants[i])
            vals = vals.real
            sorted_indices = torch.argsort(vals)
            sorted_vals = vals[sorted_indices]
            sorted_vecs = vecs[:, sorted_indices].flatten() 
            eigenvalues_tensor[i, :] = sorted_vals
            eigenvectors_tensor[i, :] = sorted_vecs.real  
        eigen=torch.cat((eigenvalues_tensor,eigenvectors_tensor), dim=1)
        force_constants = force_constants.view(-1,9)

        pair = [tuple(edge_index[:,i].tolist()) for i in range(edge_num)]
        data = Data(x=x,edge_src=edge_src+1, edge_dst=edge_dst+1, 
                    pos=pos, lattice=lattice, force_constants=force_constants,
                    force_constants_all=force_constants_all,
                    edge_attr=edge_attr , edge_vec = edge_vec, 
                    edge_index=edge_index, edge_num=edge_num, pair=pair,
                    eigenvalues=eigenvalues_tensor,eigenvectors=eigenvectors_tensor,
                    eigen=eigen)
        return data
    

class CrystalDataset(Dataset):
    def __init__(self, data_list):
        super(CrystalDataset, self).__init__()
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]



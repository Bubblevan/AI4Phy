import torch
import numpy as np
import yaml
from collections import namedtuple
from torch_geometric.data import Data
import os
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer , Module
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GCNConv , global_mean_pool ,LayerNorm , BatchNorm, GATConv
import torch.nn.init as init
from skipatom import AtomVectors

skipatom_model = AtomVectors.load("skipatom/data/mat2vec.dim200.model")
atom_embeddings = {}
for atom in skipatom_model.dictionary:
    atom_index = skipatom_model.dictionary[atom]
    atom_embedding = skipatom_model.vectors[atom_index]
    atom_embeddings[atom] = atom_embedding

atom_to_index = {atom: idx for idx, atom in enumerate(atom_embeddings)} # 生成字典，键为元素，值为序号
pretrained_embeddings = torch.tensor(list(atom_embeddings.values()))    # 生成元素嵌入矩阵

class GraphTransformer(torch.nn.Module):
    def __init__(self, node_features, num_layers, heads, pretrained_embeddings = None , embedding_dim = 200):
        super(GraphTransformer, self).__init__()
        self.conv1 = GCNConv(node_features, 256)
        self.conv2 = GCNConv(256, 256)
        self.transformer_encoder_layer = TransformerEncoderLayer(d_model = 256, nhead = heads )
        self.transformer_encoder = TransformerEncoder(self.transformer_encoder_layer, num_layers)
        try:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        except:
            self.embedding = nn.Embedding(len(atom_to_index), embedding_dim)
        self.fc1 = torch.nn.Linear(in_features = 1, out_features=node_features)
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, 1)
        self.fc4 = torch.nn.Linear(256, 9)
        # self.transformer_encoder_layer2 = TransformerEncoderLayer(d_model = 256, nhead = heads)
        # self.transformer_encoder2 = TransformerEncoder(self.transformer_encoder_layer2, num_layers )
        # self.transformer_encoder_layer3 = TransformerEncoderLayer(d_model = 128, nhead = heads )
        # self.transformer_encoder3 = TransformerEncoder(self.transformer_encoder_layer3, num_layers )
        # self.transformer_encoder_layer4 = TransformerEncoderLayer(d_model = 128, nhead = heads )
        # self.transformer_encoder4 = TransformerEncoder(self.transformer_encoder_layer4, num_layers )
        self.batchnorm1 = BatchNorm(256, eps=1e-6)
        self.batchnorm2 = BatchNorm(256, eps=1e-6)
        self.batchnorm3 = BatchNorm(256, eps=1e-6)
        self.batchnorm4 = BatchNorm(node_features, eps=1e-6)
        self.batchnorm5 = BatchNorm(embedding_dim, eps=1e-6)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        init.normal_(self.fc1.bias, std=1e-6)
        init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        init.normal_(self.fc2.bias, std=1e-6)
        init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')
        init.normal_(self.fc3.bias, std=1e-6)
        init.kaiming_normal_(self.fc4.weight, nonlinearity='relu')
        init.normal_(self.fc4.bias, std=1e-6)

    def forward(self, x, edge_index, pos, y):
        batch_indices =[[atom_to_index[element] for element in sublist] for sublist in x]
        flat_list = [item for sublist in batch_indices for item in sublist]
        x = torch.tensor(flat_list)
        x = self.embedding(x)
        #x = self.batchnorm5(x)
        edge_index = edge_index - 1
        row, col = edge_index
        edge_features = pos[col] - pos[row]
        edge_features=torch.norm(edge_features, p=2, dim=1).view(-1, 1)
        edge_features = self.fc1(edge_features) #边特征嵌入
        edge_features = self.batchnorm4(edge_features)
        x = x[row] + x[col] + edge_features
        x = F.leaky_relu_(self.conv1(x, y))
        x = self.batchnorm1(x)
        x = F.leaky_relu_(x+self.conv2(x, y))
        x = self.batchnorm2(x)
        x = F.leaky_relu_(self.transformer_encoder(x))
        x = self.batchnorm3(x)
        sym_out = self.fc3(x)
        asym_out = self.fc4(x)

        return sym_out, asym_out
    
class EdgePredictor(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, pretrained_embeddings = None , embedding_dim = 200):
        super(EdgePredictor, self).__init__()
        self.conv1 = GCNConv(3*num_node_features, 128)
        self.conv2 = GCNConv(128, 64)
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        else:
            self.embedding = nn.Embedding(len(atom_to_index), embedding_dim)
            init.normal_(self.embedding.weight, mean=0, std=1)
        self.fc1 = torch.nn.Linear(1, num_node_features)
        self.fc2 = torch.nn.Linear(64, num_edge_features)
        self.batchnorm1 = BatchNorm(128, eps=1e-6)
        self.batchnorm2 = BatchNorm(64, eps=1e-6)
        self.batchnorm3 = BatchNorm(num_node_features, eps=1e-6)
        self.batchnorm4 = BatchNorm(embedding_dim, eps=1e-6)
        self.reset_parameters()

    def reset_parameters(self):
        # 初始化全连接层
        init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        init.zeros_(self.fc1.bias)
        init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        init.zeros_(self.fc2.bias)

    def forward(self, x, edge_index , pos):
        # 节点特征提取,这里的edge_index是从1开始的，所以要减1
        edge_index = edge_index - 1
        row, col = edge_index
        edge_features = pos[col] - pos[row]
        edge_features=torch.norm(edge_features, p=2, dim=1).view(-1, 1)
        edge_features = self.fc1(edge_features)
        edge_features = self.batchnorm3(edge_features)
        # 遍历x中的每个子列表
        # 因为x是从loader中取出的，所以x是列表的列表
        batch_indices =[[atom_to_index[element] for element in sublist] for sublist in x]
        flat_list = [item for sublist in batch_indices for item in sublist]
        x = torch.tensor(flat_list)
        x = self.embedding(x)
        x = self.batchnorm4(x)
        # 边特征预测
        # 这里我们简单地对边的两个节点特征进行拼接
        x = torch.cat([x[row], x[col]], dim=1)
        x = torch.cat((x,edge_features),dim=1)
        x = F.relu(self.conv1(x, edge_index))
        x = self.batchnorm1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.batchnorm2(x)
        return self.fc2(x)
    

class CGCNN(nn.Module):
    def __init__(self, num_node_features, num_edge_features, pretrained_embeddings = None , embedding_dim = 200):
        super(CGCNN, self).__init__()
        self.conv1 = GCNConv(3 * num_node_features, 256)
        self.conv2 = GCNConv(256, 256)
        self.conv3 = GCNConv(4*256, 256)
        self.gat1 = GATConv(256, 256, heads=4, concat=True)
        self.fc1 = nn.Linear(256, 1)
        self.fc2 = nn.Linear(256, num_edge_features)
        self.fc3 = nn.Linear(3,num_node_features)
        self.batchnorm1 = BatchNorm(256, eps=1e-6)
        self.batchnorm2 = BatchNorm(256, eps=1e-6)
        self.batchnorm3 = BatchNorm(256, eps=1e-6)
        self.batchnorm4 = BatchNorm(num_node_features, eps=1e-6)
        self.batchnorm5 = BatchNorm(embedding_dim, eps=1e-6)
        self.layernorm1 = LayerNorm(256)
        self.layernorm2 = LayerNorm(4*256)
        # self.dropout = nn.Dropout(p=0.5)
        try:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        except:
            self.embedding = nn.Embedding(len(atom_to_index), embedding_dim)
            init.normal_(self.embedding.weight, mean=0, std=1)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        init.normal_(self.fc1.bias, std=1e-6)
        init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        init.normal_(self.fc2.bias, std=1e-6)
        init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')
        init.normal_(self.fc3.bias, std=1e-6)

    def forward(self, x, edge_index, pos, y):
        edge_index = edge_index - 1
        row, col = edge_index
        edge_features = (pos[col] - pos[row]).view(-1,3)
        # edge_features = torch.norm(edge_features, p=2, dim=1).view(-1, 1)
        edge_features = self.fc3(edge_features) #边特征嵌入
        edge_features = self.batchnorm4(edge_features)
        batch_indices =[[atom_to_index[element] for element in sublist] for sublist in x]

        flat_list = [item for sublist in batch_indices for item in sublist]
        x = torch.tensor(flat_list)
        x = self.embedding(x)
        x = self.batchnorm5(x)
        x = torch.cat([x[row], x[col]], dim=1)
        x = torch.cat((x,edge_features),dim=1)
        # 图卷积层
        x = F.leaky_relu_(self.conv1(x, y))
        x = self.batchnorm1(x)
        x = F.dropout(x, training=self.training)
        x = self.layernorm1(x)
        x = F.leaky_relu_(x + self.conv2(x, y))
        x = self.batchnorm2(x)
        x = self.gat1(x, y)
        x = self.layernorm2(x)
        x = F.leaky_relu_(self.conv3(x, y))
        x = self.batchnorm3(x)
        # 全局池化
        # x = global_mean_pool(x, batch)
        # 全连接层
        # x = F.relu(self.fc1(x))
        # x = self.dropout(x)

        sym_out = F.leaky_relu_(self.fc1(x))
        asym_out = F.leaky_relu_(self.fc2(x))
        return sym_out, asym_out
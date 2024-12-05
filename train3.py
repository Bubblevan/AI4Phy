import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import os
import function
import loader
import random
from module import CGCNN
from torch.optim.lr_scheduler import StepLR
if torch.cuda.is_available():
    torch.set_default_dtype(torch.float32)
    torch.set_default_device('cuda')
    torch.cuda.set_device(3)
else:
    torch.set_default_dtype(torch.float32)
    torch.set_default_device('cpu')

# gpu_list = [0,1,2,3,4,5,6,7]
# gpu_list_str = ','.join(map(str, gpu_list))
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from skipatom import AtomVectors
skipatom_model = AtomVectors.load("skipatom/data/mat2vec.dim200.model")
atom_embeddings = {}
for atom in skipatom_model.dictionary:
    atom_index = skipatom_model.dictionary[atom]
    atom_embedding = skipatom_model.vectors[atom_index]
    atom_embeddings[atom] = atom_embedding

atom_to_index = {atom: idx for idx, atom in enumerate(atom_embeddings)}
pretrained_embeddings = torch.tensor(list(atom_embeddings.values()))

directory='./data'
files_and_dirs = os.listdir(directory)

num_epochs = 500
num_node_features=200
num_edge_features=9
directory='./data'


model = CGCNN(num_node_features, 
                num_edge_features, 
                pretrained_embeddings=pretrained_embeddings)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of parameters: {total_params}")

optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
scheduler = StepLR(optimizer, step_size=1, gamma=0.8)
criterion = nn.MSELoss()
lambda_l2 = 1e-15

all_dirs = [f for f in files_and_dirs if os.path.isdir(os.path.join(directory, f))]
random.shuffle(all_dirs)
# 计算各个集合的大小
total_dirs = len(all_dirs)
train_size = int(total_dirs * 0.8)
val_size = int(total_dirs * 0.1)

train_dirs = all_dirs[:train_size]
val_dirs = all_dirs[train_size:train_size + val_size]
test_dirs = all_dirs[train_size + val_size:]

# device_ids = [1, 2, 3]
# if torch.cuda.device_count() > 1:
#     print("use", torch.cuda.device_count(), "GPUs")
#     model = torch.nn.DataParallel(model, device_ids)

with open('./TrainLog/CGCNN_training_log.txt', 'w') as file:
    pass
with open('./TrainLog/CGCNN_val.txt', 'w') as file:
    pass
with open('./TrainLog/CGCNN_training.txt', 'w') as file:
     pass

for epoch in range(num_epochs):
    model.train()
    dirs = train_dirs
    while dirs:
        # total_loss = 0
        data_loader , dirs = loader.generate_loader_cuda(dirs, directory, loader_size = 128 , batch_size = 16)
        for batch in data_loader:  # 遍历每个batch
            optimizer.zero_grad()
            # 前向传播
            sym_out, asym_out = model(batch.x, batch.edge_index, batch.pos, batch.y.t())
            edge_index = batch.edge_index - 1
            pos = batch.pos
            row, col = edge_index
            pos = pos[col] - pos[row]
            outputs = [function.construct_rotation_matrix(pos[i, :]) for i in range(len(pos))]
            # 线性变化
            sym_out = [sym_out[i]*outputs[i][:,0].view(3,1).matmul(outputs[0][:,0].view(1,3)).view(1,9) for i in range(len(pos))]
            sym_out = torch.cat(sym_out, dim=0)

            # 计算损失，这里假设是MSE损失, L2 正则化项
            l2_reg = sum(p.pow(2.0).sum() for p in model.parameters())
            loss = criterion(asym_out+sym_out, batch.edge_attr) + lambda_l2 * l2_reg
            # loss = F.mse_loss(out, batch.edge_attr)
            # 反向传播
            loss.backward()
            optimizer.step()
            # total_loss += loss.item()

            with open('./TrainLog/CGCNN_training_log.txt', 'a') as file:
                for name, parms in model.named_parameters():
                    file.write('-->name:'+ name  + ' -->grad_value:'+ str(parms.grad) + '\n')

            with open('./TrainLog/CGCNN_training.txt', 'a') as file:
                file.write(f'Epoch {epoch}, Loss: {loss.item()}'+ '\n')
        #     print(f'Epoch {epoch}, Loss: {loss.item() }')
        # print(f'Epoch {epoch}, Total_Loss: {total_loss }')
    scheduler.step()

    model.eval()
    dirs = val_dirs
    #在dirs中随机选取一个batch
    data_loader , dirs= loader.generate_loader_cuda(dirs, directory, loader_size = 128 , batch_size = 16)
    for batch in data_loader:
        sym_out, asym_out = model(batch.x, batch.edge_index, batch.pos, batch.y.t())
        edge_index = batch.edge_index - 1
        pos = batch.pos
        row, col = edge_index
        pos = pos[col] - pos[row]
        outputs = [function.construct_rotation_matrix(pos[i, :]) for i in range(len(pos))]
        # 线性变化
        sym_out = [sym_out[i]*outputs[i][:,0].view(3,1).matmul(outputs[0][:,0].view(1,3)).view(1,9) for i in range(len(pos))]
        sym_out = torch.cat(sym_out, dim=0)

        # 计算损失，这里假设是MSE损失, L2 正则化项
        l2_reg = sum(p.pow(2.0).sum() for p in model.parameters())
        loss = criterion(asym_out+sym_out, batch.edge_attr) + lambda_l2 * l2_reg

        with open('./TrainLog/CGCNN_val.txt', 'a') as file:
                file.write(f'Epoch {epoch}, Loss: {loss.item()}'+ '\n')

    if epoch % 5 == 0:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch
            }
            torch.save(checkpoint, './model/CGCNN_checkpoint.pth')
    # torch.cuda.empty_cache()
    torch.save(model, './model/CGCNN_model.pth')
    torch.save(model.state_dict(), './model/CGCNN_model_params.pth')

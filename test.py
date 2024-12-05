import torch
from datasets.graph import FC
from features.process_data import get_Path,splitdata
data_source = get_Path('datasets'+'/mp/')

fold_num = 10
train_idx,valid_idx,test_idx = splitdata(data_source,fold_num,5)

train = [data_source[i] for i in train_idx]
train_dataset = FC('datasets','train', train, 5)
from torch_geometric.loader import DataLoader
train_loader = DataLoader(train_dataset, batch_size=1, 
                shuffle=True, num_workers=0, pin_memory=True, 
                drop_last=True)
for step, data in enumerate(train_loader):
    print(data.force_constants.shape)
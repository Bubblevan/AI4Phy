import torch
import torch.nn as nn

class FrobeniusNormLoss(nn.Module):
    def __init__(self):
        super(FrobeniusNormLoss, self).__init__()
    
    def forward(self, input, target):
        return torch.mean((torch.norm(input, p='fro', dim=1)-torch.norm(target, p='fro', dim=1))**2)

class EigenLoss(nn.Module):
    def __init__(self):
        super(EigenLoss,self).__init__()

    def forward(self,input,targe):
        loss_1 = torch.mean((input[:, :3] - targe[:, :3]).pow(2))
        # 对于后9个维度，每三个维度取模求差再平方
        loss_2 = torch.mean(sum((torch.norm(input[:, 3+3*i:6+3*i], dim=1) - torch.norm(targe[:, 3+3*i:6+3*i], dim=1)).pow(2) for i in range(3)))
        # 总损失
        return (loss_1 + loss_2)
        
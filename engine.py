import torch
import torch.nn as nn
from typing import Iterable, Optional
from timm.utils import accuracy, ModelEmaV2, dispatch_clip_grad
import time
from torch_cluster import radius_graph
from concurrent.futures import ThreadPoolExecutor
import torch_geometric
from nets.loss import FrobeniusNormLoss,EigenLoss
from torch.autograd.functional import jacobian
ModelEma = ModelEmaV2

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_data(model: torch.nn.Module, criterion: torch.nn.Module,
                    x,y,batch,
                    edge_src,edge_dst,edge_vec,edge_attr,edge_num,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int,
                    amp_autocast=None,
                    loss_scaler=None,
                    clip_grad=None,
                    print_freq: int = 100,
                    logger=None,
                    force_constants=None,
                    eigen=None):
    model.train()
    criterion.train()

    mae_metric = AverageMeter()
    start_time = time.perf_counter()

    with amp_autocast():
        pred = model(batch=batch,
            f_in=x, edge_src=edge_src, edge_dst=edge_dst,edge_attr=edge_attr,
            edge_vec=edge_vec, edge_num=edge_num)
        loss = criterion(pred, eigen)
    optimizer.zero_grad()
    if loss_scaler is not None:
        loss_scaler(loss, optimizer, parameters=model.parameters())
    else:
        loss.backward()
        if clip_grad is not None:
            dispatch_clip_grad(model.parameters(),
                value=clip_grad, mode='norm')
        optimizer.step()


    mae_metric.update(torch.mean(torch.abs(pred-eigen)).item(),n=pred.shape[0])
    return mae_metric.avg

def train_one_epoch_dx(
    model: torch.nn.Module, 
    criterion: torch.nn.Module,
    data_loader: Iterable, 
    optimizer: torch.optim.Optimizer,
    device: torch.device, 
    epoch: int, 
    model_ema: Optional[ModelEma] = None,  
    amp_autocast=None,
    loss_scaler=None,
    clip_grad=None,
    print_freq: int = 50, 
    logger=None
):
    model.train()
    criterion.train()

    loss_metric = AverageMeter()
    mae_metric = AverageMeter()
    start_time = time.perf_counter()

    for step, data in enumerate(data_loader):
        data = data.to(device)

        # 打印每个batch的信息
        # print(f"Step {step}: Total nodes in data.pos: {data.pos.size(0)}")
        # print(f"Batch tensor: {data.batch}")
        # print(f"Unique samples in batch: {data.batch.unique()}")

        with amp_autocast():
            # Model forward pass
            pred = model(
                batch=data.batch, 
                f_in=data.x, 
                edge_src=data.edge_src, 
                edge_dst=data.edge_dst,
                pos=data.pos, 
                edge_num=data.edge_num, 
                device=device
            )
            pred = pred.squeeze()  # Shape: [batch_size,]

            # First-order gradients
            grad_outputs = torch.ones_like(pred)  # Shape: [batch_size,]
            grads = torch.autograd.grad(
                pred, data.pos, 
                grad_outputs=grad_outputs, 
                create_graph=True, 
                retain_graph=True, 
                allow_unused=True
            )[0]

            # Hessian computation: Separate by idx
            batch_hessians = []
            for sample_idx in data.batch.unique():  # Process each graph separately
                sample_mask = data.batch == sample_idx
                sample_pos = data.pos[sample_mask]
                sample_grads = grads[sample_mask]

                n = sample_pos.size(0)

                # Hessian computation for the current graph
                hessian = torch.zeros(n, n, 3, 3, requires_grad=True).to(device)

                # Vectorized Hessian computation
                for k in range(3):  # x, y, z components
                    grad_outputs = torch.zeros_like(sample_grads)
                    grad_outputs[:, k] = 1.0
                    grad_grad = torch.autograd.grad(
                        sample_grads, sample_pos,
                        grad_outputs=grad_outputs,
                        create_graph=False, 
                        retain_graph=True, 
                        allow_unused=True
                    )[0]

                    if grad_grad is not None:
                        # Fill Hessian tensor for the current dimension
                        hessian[:, :, :, k] = grad_grad.unsqueeze(-1)

                hessian = hessian.view(-1, 9)  # Flatten Hessian for current graph
                batch_hessians.append(hessian)

            # Combine Hessians
            hessian = torch.cat(batch_hessians, dim=0)

            # Ensure shapes match for loss computation
            if hessian.shape != data.force_constants_all.shape:
                raise ValueError(
                    f"Dimension mismatch: hessian shape {hessian.shape}, "
                    f"target shape {data.force_constants_all.shape}"
                )

            # Compute the loss
            loss = criterion(hessian, data.force_constants_all)

        # Backpropagation
        optimizer.zero_grad()

        if loss_scaler is not None:
            loss_scaler(loss, optimizer, parameters=model.parameters())
        else:
            loss.backward()
            if clip_grad is not None:
                dispatch_clip_grad(model.parameters(), value=clip_grad, mode='norm')
            optimizer.step()

        # Metrics
        loss_metric.update(loss.item(), n=hessian.shape[0])
        mae_metric.update(
            torch.mean(torch.abs(hessian - data.force_constants_all)).item(), 
            n=hessian.shape[0]
        )

        # Logging
        if step % print_freq == 0 and logger:
            logger.info(f"Epoch [{epoch}], Step [{step}/{len(data_loader)}], "
                        f"Loss: {loss_metric.avg:.4f}, MAE: {mae_metric.avg:.4f}")

    return mae_metric.avg, loss_metric.avg

def evaluate_dx(model, data_loader, device, amp_autocast=None, 
                print_freq=100, logger=None):
    
    print_freq = 8
    model.eval()
    
    loss_metric = AverageMeter()
    mae_metric = AverageMeter()
    criterion = nn.MSELoss()
    criterion.eval()
    
    for step, data in enumerate(data_loader):
        data = data.to(device)
        with amp_autocast():
            pred = model(batch=data.batch, 
                         f_in=data.x, edge_src=data.edge_src, edge_dst=data.edge_dst, 
                         pos=data.pos, edge_num=data.edge_num, device=device)
            
            grad = torch.autograd.grad(pred, data.pos, 
                                       create_graph=True, retain_graph=True)[0]
            
            # 0.6120238304138184 GB
            # print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024 ** 3} GB") 
            # print(f"Memory Cached: {torch.cuda.memory_cached(0) / 1024 ** 3} GB")
            
            n = data.pos.size(0)
            hessian = torch.zeros(n, n, 3, 3, requires_grad=True).to(device)
            
            # 1.0705695152282715 GB
            for i in range(n):
                grads = torch.autograd.grad(pred, data.pos, create_graph=True, retain_graph=True)[0]
                for k in range(3):
                    grad_outputs = torch.zeros_like(grads)
                    grad_outputs[i, k] = 1.0
                    grad_grad = torch.autograd.grad(grads, data.pos, grad_outputs=grad_outputs, create_graph=False, retain_graph=True)[0]
                    for j in range(n):
                        for l in range(3):
                            hessian[i, j, k, l] = grad_grad[j, l]
                    torch.cuda.empty_cache()
            
            hessian = hessian.view(-1, 9)
            loss = criterion(hessian, data.force_constants_all)
        
        loss_metric.update(torch.mean(loss).item(), n=hessian.shape[0])
        mae_metric.update(torch.mean(torch.abs(hessian - data.force_constants_all)).item(), n=hessian.shape[0])

        # if step % print_freq == 0 or step == len(data_loader) - 1: #time.perf_counter() - wall_print > 15:
        #     info_str = '[{step}/{length}] loss: {loss:.5f}, MAE: {mae:.5f}, '.format( 
        #         step=step, length=len(data_loader), 
        #         mae=mae_metric.avg, 
        #         loss=loss_metric.avg,
        #         )
        #     logger.info(info_str)
    
    return mae_metric.avg, loss_metric.avg

def compute_stats(data_loader, max_radius, logger, print_freq=1000):
    '''
        Compute mean of numbers of nodes and edges
    '''
    log_str = '\nCalculating statistics with '
    log_str = log_str + 'max_radius={}\n'.format(max_radius)
    logger.info(log_str)
        
    avg_node = AverageMeter()
    avg_edge = AverageMeter()
    avg_degree = AverageMeter()
    
    for step, data in enumerate(data_loader):
        
        pos = data.pos
        batch = data.batch
        edge_src, edge_dst = radius_graph(pos, r=max_radius, batch=batch,
            max_num_neighbors=1000)
        batch_size = float(batch.max() + 1)
        num_nodes = pos.shape[0]
        num_edges = edge_src.shape[0]
        num_degree = torch_geometric.utils.degree(edge_src, num_nodes)
        num_degree = torch.sum(num_degree)
            
        avg_node.update(num_nodes / batch_size, batch_size)
        avg_edge.update(num_edges / batch_size, batch_size)
        avg_degree.update(num_degree / (num_nodes), num_nodes)
            
        if step % print_freq == 0 or step == (len(data_loader) - 1):
            log_str = '[{}/{}]\tavg node: {}, '.format(step, len(data_loader), avg_node.avg)
            log_str += 'avg edge: {}, '.format(avg_edge.avg)
            log_str += 'avg degree: {}, '.format(avg_degree.avg)
            logger.info(log_str)


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, 
                    model_ema: Optional[ModelEma] = None,  
                    amp_autocast=None,
                    loss_scaler=None,
                    clip_grad=None,
                    print_freq: int = 100, 
                    logger=None):
    
    model.train()
    criterion.train()
    
    loss_metric = AverageMeter()
    mae_metric = AverageMeter()

    start_time = time.perf_counter()
    
    for step, data in enumerate(data_loader):
        data = data.to(device)
        with amp_autocast():
            pred = model(batch=data.batch, 
                f_in=data.x, edge_src=data.edge_src, edge_dst=data.edge_dst,edge_attr=data.edge_attr,
                edge_vec=data.edge_vec, edge_num = data.edge_num)

            loss = criterion(pred, data.force_constants)
        
        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(loss, optimizer, parameters=model.parameters())
        else:
            loss.backward()
            if clip_grad is not None:
                dispatch_clip_grad(model.parameters(), 
                    value=clip_grad, mode='norm')
            optimizer.step()
        
        loss_metric.update(loss.item(), n=pred.shape[0])

        mae_metric.update(torch.mean(torch.abs(pred-data.force_constants)).item(), n=pred.shape[0])

        if model_ema is not None:
            model_ema.update(model)
        
        torch.cuda.synchronize()
        
        # logging
        if step % print_freq == 0 or step == len(data_loader) - 1: #time.perf_counter() - wall_print > 15:
            w = time.perf_counter() - start_time
            e = (step + 1) / len(data_loader)
            info_str = 'Epoch: [{epoch}][{step}/{length}] loss: {loss:.5f}, MAE: {mae:.5f}, time/step={time_per_step:.0f}ms, '.format( 
                epoch=epoch, step=step, length=len(data_loader), 
                mae=mae_metric.avg, 
                loss=loss_metric.avg,
                time_per_step=(1e3 * w / e / len(data_loader))
                )
            info_str += 'lr={:.2e}'.format(optimizer.param_groups[0]["lr"])
            logger.info(info_str)
    return mae_metric.avg , loss_metric.avg

def evaluate(model, data_loader, device, amp_autocast=None, 
    print_freq=100, logger=None):
    
    model.eval()
    
    loss_metric = AverageMeter()
    mae_metric = AverageMeter()
    criterion = torch.nn.L1Loss()
    criterion.eval()
    
    with torch.no_grad():
            
        for data in data_loader:
            data = data.to(device)
            with amp_autocast():
                pred = model(batch=data.batch, 
                f_in=data.x, edge_src=data.edge_src, edge_dst=data.edge_dst,edge_attr=data.edge_attr,
                edge_vec=data.edge_vec, edge_num = data.edge_num)
                loss = criterion(pred, data.force_constants)

            loss_metric.update(loss.item(), n=pred.shape[0])
            mae_metric.update(torch.mean(torch.abs(pred-data.force_constants)).item(), n=pred.shape[0])

    return mae_metric.avg, loss_metric.avg


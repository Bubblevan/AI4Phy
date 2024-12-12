import torch
import torch.profiler
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn as nn
from typing import Iterable, Optional
from timm.utils import accuracy, ModelEmaV2, dispatch_clip_grad
import time
from torch_cluster import radius_graph
from concurrent.futures import ThreadPoolExecutor
import torch_geometric
from nets.loss import FrobeniusNormLoss,EigenLoss
from torch.autograd.functional import jacobian, hessian
# from torchviz import make_dot
import logging

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

def compute_hessian_vectorized(model, inputs, batch, f_in, edge_src, edge_dst, edge_num, device):
    def loss_fn(positions):
        # 根据当前 positions 重新预测
        pred = model(
            batch=batch,
            f_in=f_in,
            edge_src=edge_src,
            edge_dst=edge_dst,
            pos=positions,
            edge_num=edge_num,
            device=device,
        )
        return pred.sum()

    # 计算 Hessian, 返回形状为 [N,3,N,3] 的张量
    hessian_matrix = hessian(loss_fn, inputs)  # 可能返回 (N*3, N*3)
    # 若 hessian 返回为 (N*3, N*3)，则需要 reshape 成 (N,3,N,3)
    N = inputs.size(0)
    hessian_matrix = hessian_matrix.view(N, 3, N, 3)
    return hessian_matrix

def trace_grad_fn(tensor, visited=None):
    if visited is None:
        visited = set()

    # 如果当前张量是叶子节点，直接返回
    if tensor.grad_fn is None:
        print(f"Reached a leaf tensor: {tensor.shape}")
        return

    # 如果当前 grad_fn 已经被访问过，检测到循环
    if tensor.grad_fn in visited:
        print(f"Detected a cycle in the computation graph at grad_fn: {tensor.grad_fn}")
        return

    # 打印当前张量和 grad_fn
    print(f"Current tensor: {tensor.shape}, grad_fn: {tensor.grad_fn}")
    visited.add(tensor.grad_fn)

    # 递归遍历 next_functions
    for next_fn, _ in tensor.grad_fn.next_functions:
        if next_fn is not None:
            # 尝试获取 next_fn 的输入张量
            try:
                # 获取 next_fn 的输入张量
                input_tensors = next_fn.next_inputs()
                for input_tensor in input_tensors:
                    trace_grad_fn(input_tensor, visited)
            except AttributeError:
                # 如果 next_fn 没有 next_inputs 方法，跳过
                continue

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
    # 保存初始模型参数
    before_params = [param.clone().detach() for param in model.parameters()]

    model.train()
    criterion.train()

    loss_metric = AverageMeter()
    mae_metric = AverageMeter()
    print("Inspecting data_loader contents:")
    for idx, batch in enumerate(data_loader):
        print(f"Batch {idx}:")
        print(f"  Batch data.batch.unique() = {batch.batch.unique()}")
        print(f"  Batch pos.shape = {batch.pos.shape}")
        # if idx >= 2:  # 仅打印前三个 batch，避免过多输出
        #     break
    for step, data in enumerate(data_loader):
        logger.info(f"Processing step {step}/{len(data_loader)}")
        data = data.to(device)
        data.pos.requires_grad_(True)

        with amp_autocast():

            # pred = model(
            #     batch=data.batch, 
            #     f_in=data.x, 
            #     edge_src=data.edge_src, 
            #     edge_dst=data.edge_dst,
            #     pos=data.pos, 
            #     edge_num=data.edge_num, 
            #     device=device
            # )
            # pred = pred.squeeze() # shape: [batchsize, ]
            # grad_outputs = torch.ones_like(pred, device=device)
            # grads = torch.autograd.grad(pred, data.pos, grad_outputs=grad_outputs, create_graph=True, retain_graph=True)[0]
            print(f"data.batch: {data.batch}")
            print(f"data.batch.unique(): {data.batch.unique()}")
            batch_hessians = []
            for sample_idx in data.batch.unique():
                sample_mask = data.batch == sample_idx
                sample_pos = data.pos[sample_mask]
                sample_pos.requires_grad_(True)
                print(f"Sample {sample_idx}: pos.shape = {sample_pos.shape}")

                # sample_grads = grads[sample_mask]
                logger.info("Starting model forward pass.")

                # sample_pred = model(
                #     batch=data.batch[sample_mask],
                #     f_in=data.x[sample_mask],
                #     edge_src=data.edge_src,
                #     edge_dst=data.edge_dst,
                #     pos=sample_pos,
                #     edge_num=data.edge_num,
                #     device=device,
                # )
                # print("看看sample_pred的gradfn")
                # print(sample_pred.grad)
                # trace_grad_fn(sample_pred)
                # sample_pred = sample_pred.squeeze()
                # sample_pred.requires_grad_(True)
                # trace_grad_fn(sample_pred)

                logger.info("Starting gradient computation.")
                # grad_outputs = torch.ones_like(sample_pred, device=device, requires_grad=True)
                # sample_grads = torch.autograd.grad(
                #     sample_pred, sample_pos, 
                #     grad_outputs=grad_outputs, 
                #     create_graph=True, 
                #     retain_graph=True, 
                #     allow_unused=False  # 检查计算图连接是否正确
                # )[0]
                # print(sample_grads)
                # print(f"grad_outputs.shape: {grad_outputs.shape}")
                # print(f"sample_grads.shape: {sample_grads.shape}")
                # print("看看一阶导sample_grads的gradfn")
                # print(sample_grads.grad)
                # trace_grad_fn(sample_grads)
                # print(f"sample_grads.requires_grad: {sample_grads.requires_grad}")
                # print(f"sample_pos.requires_grad: {sample_pos.requires_grad}")

                n = sample_pos.size(0)
                print(f"逐样本处理的这个样本的原子数为: {n}")
                def model_pred(pos):
                    sample_pred = model(
                        batch=data.batch[sample_mask],
                        f_in=data.x[sample_mask],
                        edge_src=data.edge_src,
                        edge_dst=data.edge_dst,
                        pos=pos,
                        edge_num=data.edge_num,
                        device=device,
                    )
                    return sample_pred.sum()

                logger.info("Starting Hessian computation using functional API.")
                hessian = torch.autograd.functional.hessian(model_pred, sample_pos, create_graph=True)
                # hessian = torch.autograd.functional.hessian(
                #     lambda pos: model(
                #         batch=data.batch[sample_mask],
                #         f_in=data.x[sample_mask],
                #         edge_src=data.edge_src,
                #         edge_dst=data.edge_dst,
                #         pos=sample_pos,
                #         edge_num=data.edge_num,
                #         device=device,
                #     ).squeeze(),
                #     sample_pos,
                #     create_graph=True
                # )
                # hessian = torch.zeros(n, n, 3, 3, requires_grad=True).to(device) 
                # for k in range(3):  # x, y, z components
                #     grad_outputs = torch.zeros_like(sample_grads)
                #     grad_outputs[:, k] = 1.0
                #     grad1 = torch.autograd.grad(
                #         sample_grads, sample_pos,
                #         grad_outputs=grad_outputs,
                #         create_graph=True,
                #         retain_graph=True,
                #         allow_unused=False
                #     )[0]
                #     if grad1 is not None:
                #         hessian[:, :, :, k] = grad1.unsqueeze(1) 
                hessian = hessian.view(n, 3, n, 3).permute(0, 2, 1, 3).contiguous()
                hessian = hessian.view(-1, 9)

                num_samples = 10  # 选择 10 个随机值
                indices = torch.randperm(hessian.numel())[:num_samples]
                selected_values = hessian.view(-1)[indices]
                selected_values_str = "\n".join([f"{value.item():.4e}" for value in selected_values])
                logger.info("Sampled Hessian values:\n%s", selected_values_str)
                print(f"data.pos.grad: {data.pos.grad}")
                print("看看单个样本的hessian的gradfn：")
                trace_grad_fn(hessian)

                batch_hessians.append(hessian)

            hessian_final = torch.cat(batch_hessians, dim=0)
            hessian_final.requires_grad_(True)

            # # 计算整个batch的 Hessian
            # hessian_matrix = torch.autograd.functional.hessian(loss_fn, data.pos)
            # # hessian_matrix = torch.autograd.grad(grads, data.pos, grad_outputs=grad_outputs, create_graph=True, retain_graph=True)[0]
            # N = data.pos.size(0)
            # hessian_matrix = hessian_matrix.view(N, 3, N, 3)

            # # 根据 batch 中每个图的节点进行分块匹配 target
            # # data.batch: [N], 包含每个节点所属的图 id
            # unique_graphs = data.batch.unique()  
            # hessian_list = []
            # node_count_list = []
            # if logger:
            #     logger.info("Processing Hessian matrix for subgraphs.")
            # for g_id in unique_graphs:
            #     # 该图的节点mask
            #     nodes_g = (data.batch == g_id)
            #     # 提取子图的 Hessian 子块
            #     sub_hessian = hessian_matrix[nodes_g][:, :, nodes_g, :]
            #     # sub_hessian shape: (n_g,3,n_g,3)
            #     n_g = sub_hessian.size(0)
            #     # 重塑为 (n_g^2, 9)
            #     sub_hessian = sub_hessian.reshape(n_g * n_g, 9)
            #     hessian_list.append(sub_hessian)
            #     node_count_list.append(n_g)

            # # 将所有子图的 Hessian 拼接，形成 (sum(n_g^2), 9) 的张量
            # hessian_final = torch.cat(hessian_list, dim=0)

            # force_constants_all 应该同样是按照每个子图的 Hessian 块依次排列
            # 确保 target 和 hessian_final 的维度对应。假设 data.force_constants_all
            # 已经是 (sum(n_g^2), 9) 的形状。
            if hessian_final.shape != data.force_constants_all.shape:
                raise ValueError(
                    f"Dimension mismatch: hessian shape {hessian_final.shape}, "
                    f"target shape {data.force_constants_all.shape}"
                )
                
            data.force_constants_all = data.force_constants_all.to(device)
            data.force_constants_all.requires_grad_(True)
            print("看看hessian_final的gradfn：")
            trace_grad_fn(hessian_final)

            # 计算损失
            if logger:
                logger.info("Computing loss.")
            loss = criterion(hessian_final, data.force_constants_all).requires_grad_()
            logger.info(f"loss的值：{loss}")
            logger.info(f"Loss grad_fn: {loss.grad_fn}, requires_grad: {loss.requires_grad}")
            trace_grad_fn(loss)


            logger.info("Visualizing computation graph.")
            # dot = make_dot(loss, params={"data.pos": data.pos})
            # dot.render(f"computation_graph_step_{step}", format="pdf")        

        optimizer.zero_grad()

        if loss_scaler is not None:
            loss_scaler(loss, optimizer, parameters=model.parameters())
        else:
            if logger:
                logger.info("Performing backward pass and optimizer step.")
            loss.backward()
            print(f"data.pos.grad: {data.pos.grad}")
            # 检查模型参数的梯度是否存在非零值
            # 如果所有梯度为零或 None，说明梯度未正确反向传播

            for name, param in model.named_parameters():
                # logger.info(f"{name} requires_grad: {param.requires_grad}")
                if param.grad is not None:
                    logger.info(f"{name} gradient norm: {param.grad.norm().item()}")
                # else:
                    # logger.info(f"{name} has no gradient!")

            if clip_grad is not None:
                dispatch_clip_grad(model.parameters(), value=clip_grad, mode='norm')
            
            optimizer.step()

            # 检查优化器状态
            for group in optimizer.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        logger.info(f"Optimizer state for param {id(p)}:", optimizer.state[p])
            
            # 保存更新后的参数
            after_params = [param.clone().detach() for param in model.parameters()]
            # 比较参数是否发生变化
            for idx, (before, after) in enumerate(zip(before_params, after_params)):
                if not torch.equal(before, after):
                    logger.info(f"参数已更新: 参数索引 {idx}")
                    param_name = list(model.state_dict().keys())[idx]
                    logger.info(f"更新的参数名称: {param_name}")
                    logger.info(f"更新前的参数值: {before}")
                    logger.info(f"更新后的参数值: {after}")
                    break
            else:
                logger.info("参数未更新")
            before_params = after_params


        # 记录损失和MAE
        loss_metric.update(loss.item(), n=hessian_final.shape[0])
        mae_metric.update(
            torch.mean(torch.abs(hessian_final - data.force_constants_all)).item(), 
            n=hessian_final.shape[0]
        )

        if step % print_freq == 0 and logger is not None:
            logger.info(
                f"Epoch [{epoch}], Step [{step}/{len(data_loader)}], "
                f"Loss: {loss_metric.avg:.4f}, MAE: {mae_metric.avg:.4f}"
            )

    return mae_metric.avg, loss_metric.avg


def evaluate_dx(model, data_loader, device, amp_autocast=None, print_freq=100, logger=None):
    model.eval()
    
    loss_metric = AverageMeter()
    mae_metric = AverageMeter()
    criterion = nn.MSELoss()
    criterion.eval()
    
    for step, data in enumerate(data_loader):
        data = data.to(device)
        
        # 确保 data.pos 需要梯度
        data.pos.requires_grad_(True)
        
        with amp_autocast():
            # 前向计算
            pred = model(
                batch=data.batch,
                f_in=data.x,
                edge_src=data.edge_src,
                edge_dst=data.edge_dst,
                pos=data.pos,
                edge_num=data.edge_num,
                device=device
            )
            pred = pred.view(-1)

            # 一阶梯度
            grad_outputs = torch.ones_like(pred)
            grads = torch.autograd.grad(
                pred, data.pos, grad_outputs=grad_outputs, create_graph=True, retain_graph=True
            )[0]

            # 整批 Hessian 计算
            hessian_matrix = compute_hessian_vectorized(
                model=model,
                inputs=data.pos,
                batch=data.batch,
                f_in=data.x,
                edge_src=data.edge_src,
                edge_dst=data.edge_dst,
                edge_num=data.edge_num,
                device=device,
            )  # shape: [N,3,N,3]

        # 根据 batch 的子图划分 Hessian
        unique_graphs = data.batch.unique()
        hessian_list = []
        for g_id in unique_graphs:
            nodes_g = (data.batch == g_id)
            # 提取该图子块 (n_g,3,n_g,3)
            sub_hessian = hessian_matrix[nodes_g][:, :, nodes_g, :]
            n_g = sub_hessian.size(0)
            # 重塑成 (n_g², 9)
            sub_hessian = sub_hessian.reshape(n_g * n_g, 9)
            hessian_list.append(sub_hessian)

        # 拼接所有子图的 Hessian 得到 (sum of n_g², 9)
        hessian = torch.cat(hessian_list, dim=0)

        # 计算损失和MAE
        loss = criterion(hessian, data.force_constants_all)
        mae = torch.mean(torch.abs(hessian - data.force_constants_all))

        loss_metric.update(loss.item(), n=hessian.shape[0])
        mae_metric.update(mae.item(), n=hessian.shape[0])

        # 打印日志
        if step % print_freq == 0 and logger:
            logger.info(
                f"Step [{step}/{len(data_loader)}], "
                f"Loss: {loss_metric.avg:.4f}, MAE: {mae_metric.avg:.4f}"
            )

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

def train_one_epoch_adam(
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
    logger=None,
    beta2: float = 0.999,
    eps: float = 1e-8
):
    model.train()
    criterion.train()

    loss_metric = AverageMeter()
    mae_metric = AverageMeter()

    vt_dict = {}  # 用于存储每个数据的`vt`

    for step, data in enumerate(data_loader):
        data = data.to(device)

        with amp_autocast():
            data.pos.requires_grad_(True)
            pred = model(
                batch=data.batch, 
                f_in=data.x, 
                edge_src=data.edge_src, 
                edge_dst=data.edge_dst,
                pos=data.pos, 
                edge_num=data.edge_num, 
                device=device
            )
            pred = pred.view(-1)  

            grad_outputs = torch.ones_like(pred)
            grads = torch.autograd.grad(
                pred, data.pos, grad_outputs=grad_outputs, create_graph=True, retain_graph=True
            )[0]

            # 初始化 vt 如果尚未存储
            if step == 0 and data.pos.grad_fn is not None:
                vt_dict = {name: torch.zeros_like(param) for name, param in model.named_parameters()}

            # 更新 vt
            for name, param in model.named_parameters():
                if param.grad is not None:
                    vt_dict[name] = beta2 * vt_dict[name] + (1 - beta2) * param.grad.pow(2)

            # 近似Hessian对角线
            approx_hessian = {name: vt.sqrt() + eps for name, vt in vt_dict.items()}

            # 计算损失
            loss = criterion(approx_hessian, data.force_constants_all)

        optimizer.zero_grad()

        if loss_scaler is not None:
            loss_scaler(loss, optimizer, parameters=model.parameters())
        else:
            loss.backward()
            if clip_grad is not None:
                dispatch_clip_grad(model.parameters(), value=clip_grad, mode='norm')
            optimizer.step()

        # 记录损失和MAE
        loss_metric.update(loss.item(), n=1)
        mae_metric.update(
            torch.mean(torch.abs(torch.cat(list(approx_hessian.values())) - data.force_constants_all)).item(), 
            n=1
        )

        if step % print_freq == 0 and logger is not None:
            logger.info(
                f"Train Epoch [{epoch}], Step [{step}/{len(data_loader)}], "
                f"Loss: {loss_metric.avg:.4f}, MAE: {mae_metric.avg:.4f}"
            )

    return mae_metric.avg, loss_metric.avg

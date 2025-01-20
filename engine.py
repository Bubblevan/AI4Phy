import torch
import os
import psutil
import torch.profiler
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch.nn import Module
from typing import Iterable, Optional
from timm.utils import accuracy, ModelEmaV2, dispatch_clip_grad
import time
from torch_cluster import radius_graph
from concurrent.futures import ThreadPoolExecutor
import torch_geometric
from nets.loss import FrobeniusNormLoss,EigenLoss
from torch.autograd.functional import jacobian, hessian, vjp
from torch.optim import Optimizer
# from torchviz import make_dot
import logging
from functools import partial
import random

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

def local_hessian(func, inputs, create_graph=False, strict=False, outer_jacobian_strategy="reverse-mode"):
    def _as_tuple(inp, arg_name=None, fn_name=None):
        is_inp_tuple = True
        if not isinstance(inp, tuple):
            inp = (inp,)
            is_inp_tuple = False

        for i, el in enumerate(inp):
            if not isinstance(el, torch.Tensor):
                raise TypeError(
                    f"The {arg_name} given to {fn_name} must be either a Tensor or a tuple of Tensors but the"
                    f" value at index {i} has type {type(el)}."
                )

        return is_inp_tuple, inp

    def _tuple_postprocess(res, to_unpack):
        if isinstance(to_unpack, tuple):
            if not to_unpack[1]:
                res = tuple(el[0] for el in res)
            if not to_unpack[0]:
                res = res[0]
        else:
            if not to_unpack:
                res = res[0]
        return res

    def _grad_preprocess(inputs, create_graph, need_graph):
        res = []
        for inp in inputs:
            if create_graph and inp.requires_grad:
                if not inp.is_sparse:
                    res.append(inp.view_as(inp))
                else:
                    res.append(inp.clone())
            else:
                res.append(inp.detach().requires_grad_(need_graph))
        return tuple(res)

    def ensure_single_output_function(*inp):
        out = func(*inp)
        is_out_tuple, t_out = _as_tuple(out, "outputs of the user-provided function", "hessian")
        if is_out_tuple or not isinstance(out, torch.Tensor):
            raise RuntimeError(
                "The function given to hessian should return a single Tensor"
            )
        if out.nelement() != 1:
            raise RuntimeError(
                "The Tensor returned by the function given to hessian should contain a single element"
            )
        return out.squeeze()

    def jacobian(func, inputs, create_graph=False, strict=False, strategy="reverse-mode"):
        with torch.enable_grad():
            is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "jacobian")
            inputs = _grad_preprocess(inputs, create_graph=create_graph, need_graph=True)
            outputs = func(*inputs)
            is_outputs_tuple, outputs = _as_tuple(outputs, "outputs", "jacobian")
            jacobian_result = tuple()

            for i, output_tensor in enumerate(outputs):
                grad_matrix = tuple([] for _ in range(len(inputs)))
                for output_index in range(output_tensor.nelement()):
                    grad_component = torch.autograd.grad(
                        (output_tensor.reshape(-1)[output_index],),
                        inputs,
                        retain_graph=True,
                        create_graph=create_graph,
                    )

                    for idx, (grad_list, grad_value, input_tensor) in enumerate(
                        zip(grad_matrix, grad_component, inputs)
                    ):
                        if grad_value is not None:
                            grad_list.append(grad_value)
                        else:
                            grad_list.append(torch.zeros_like(input_tensor))

                jacobian_result += (
                    tuple(
                        torch.stack(grad_list, dim=0).view(
                            output_tensor.size() + inputs[idx].size()
                        )
                        for idx, grad_list in enumerate(grad_matrix)
                    ),
                )
            return _tuple_postprocess(jacobian_result, (is_outputs_tuple, is_inputs_tuple))

    def jac_func(*inp):
        inp = tuple(t.requires_grad_(True) for t in inp)
        return jacobian(ensure_single_output_function, inp, create_graph=True)

    is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "hessian")
    result = jacobian(
        jac_func,
        inputs,
        create_graph=create_graph,
        strict=strict,
        strategy=outer_jacobian_strategy,
    )
    return _tuple_postprocess(result, (is_inputs_tuple, is_inputs_tuple))

def local_ckpt(model, x):
    """
    Compute the Hessian matrix with checkpointing to reduce memory usage.

    Args:
        model: The model or function to compute the Hessian for.
        x: The input tensor to the model.

    Returns:
        The Hessian matrix.
    """
    def forward_fn(x):
        return model(x)

    # Use checkpointing only for the forward computation
    energy = checkpoint(forward_fn, x)

    # Compute the gradient without checkpointing to avoid incompatibility
    grad = torch.autograd.grad(energy, x, create_graph=True, retain_graph=True)[0]

    # Compute the Hessian without checkpointing
    hessian = torch.autograd.functional.hessian(lambda inp: model(inp).sum(), x, create_graph=True)

    return hessian


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

    for step, data in enumerate(data_loader):
        logger.info(f"Processing step {step}/{len(data_loader)}")
        data = data.to(device)
        data.pos.requires_grad_(True)

        with amp_autocast():
            batch_hessians = []
            for sample_idx in data.batch.unique():
                sample_mask = data.batch == sample_idx
                sample_pos = data.pos[sample_mask]
                sample_pos.requires_grad_(True)
                logger.info(f"Sample {sample_idx}: pos.shape = {sample_pos.shape}")

                n = sample_pos.size(0)
                logger.info(f"Processing sample with {n} atoms.")

                # 计算力 (一阶导数)
                energy = model(
                    batch=data.batch[sample_mask],
                    f_in=data.x[sample_mask],
                    edge_src=data.edge_src,
                    edge_dst=data.edge_dst,
                    pos=sample_pos,
                    edge_num=data.edge_num,
                    device=device,
                ).sum()
                trace_grad_fn(energy)
                forces = -torch.autograd.grad(
                    energy, sample_pos, create_graph=True
                )[0]
                logger.info("计算力 (一阶导数)完成")
                trace_grad_fn(forces)
                force_constants = torch.zeros(
                    (n, n, 3, 3), device=sample_pos.device
                )

                force_constants_list = []

                for i in range(n):
                    force_constants_i = []
                    for alpha in range(3):
                        grad = torch.autograd.grad(
                            forces[i, alpha], sample_pos, retain_graph=True, create_graph=True
                        )[0]
                        force_constants_i.append(grad)
                    force_constants_list.append(torch.stack(force_constants_i, dim=1))

                # 将结果拼接成张量
                force_constants = torch.stack(force_constants_list, dim=0)

                trace_grad_fn(force_constants)
                logger.info("初始化 Hessian 矩阵 (二阶导数)完成")
                # 重塑并保存结果
                hessian = force_constants.view(n, n, 3, 3).permute(0, 2, 1, 3).contiguous()
                trace_grad_fn(hessian)
                hessian = hessian.view(-1, 9)
                trace_grad_fn(hessian)
                batch_hessians.append(hessian)

            # 合并 Hessian
            hessian_final = torch.cat(batch_hessians, dim=0)
            trace_grad_fn(hessian_final)

            # 目标值匹配
            if hessian_final.shape != data.force_constants_all.shape:
                raise ValueError(
                    f"Dimension mismatch: hessian shape {hessian_final.shape}, "
                    f"target shape {data.force_constants_all.shape}"
                )
            data.force_constants_all = data.force_constants_all.to(device)

            # 计算损失
            logger.info("Computing loss.")
            loss = criterion(hessian_final, data.force_constants_all)

        optimizer.zero_grad()

        if loss_scaler is not None:
            loss_scaler(loss, optimizer, parameters=model.parameters())
        else:
            loss.backward()
            if clip_grad is not None:
                dispatch_clip_grad(model.parameters(), value=clip_grad, mode='norm')
            optimizer.step()

        # 保存更新后的参数
        after_params = [param.clone().detach() for param in model.parameters()]
        for idx, (before, after) in enumerate(zip(before_params, after_params)):
            if not torch.equal(before, after):
                logger.info(f"Parameters updated at index {idx}.")
                break
        before_params = after_params

        # 记录损失和 MAE
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

def train_one_epoch_hessian(
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
        break

    # 开始 PyTorch Profiler
    # with torch.profiler.profile(
    #     activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    #     record_shapes=True,
    #     with_stack=True
    # ) as prof:
        
    for step, data in enumerate(data_loader):
        logger.info(f"Processing step {step}/{len(data_loader)}")
        data = data.to(device)
        data.pos.requires_grad_(True)

        with amp_autocast():
            print(f"data.batch: {data.batch}")
            print(f"data.batch.unique(): {data.batch.unique()}")
            batch_hessians = []
            for sample_idx in data.batch.unique():
                sample_mask = data.batch == sample_idx
                sample_pos = data.pos[sample_mask]
                sample_pos.requires_grad_(True)
                print(f"Sample {sample_idx}: pos.shape = {sample_pos.shape}")

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
                hessian = local_ckpt(model_pred, sample_pos)
                print("Hessian shape before view:", hessian.shape)

                # 将 Hessian 从 (3N, 3N) 转换为 (n, 3, n, 3)
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

            for name, param in model.named_parameters():
                if param.grad is not None:
                    logger.info(f"{name} gradient norm: {param.grad.norm().item()}")

            if clip_grad is not None:
                dispatch_clip_grad(model.parameters(), value=clip_grad, mode='norm')

            optimizer.step()

            del hessian, batch_hessians  # 删除无用的变量
            torch.cuda.empty_cache()  # 清理显存中的无用缓存

            # 保存更新后的参数
            after_params = [param.clone().detach() for param in model.parameters()]
            for idx, (before, after) in enumerate(zip(before_params, after_params)):
                if not torch.equal(before, after):
                    logger.info(f"参数已更新: 参数索引 {idx}")
                    param_name = list(model.state_dict().keys())[idx]
                    logger.info(f"更新的参数名称: {param_name}")
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

    # 打印训练过程的性能分析结果
    # prof.export_chrome_trace(f"trace_epoch_{epoch}.json")
    # prof.key_averages().table(sort_by="self_cpu_time_total")

    return mae_metric.avg, loss_metric.avg

def compute_hessian_finite_difference(model, data, epsilon=1e-6, device='cuda', threshold=10):
    n_atoms = data.pos.size(0)
    n_coords = n_atoms * 3
    hessian = torch.zeros((n_coords, n_coords), device=device)

    def energy_func(pos):
        pos = pos.view_as(data.pos)
        pred = model(
            batch=data.batch,
            f_in=data.x,
            edge_src=data.edge_src,
            edge_dst=data.edge_dst,
            pos=pos,
            edge_num=data.edge_num,
            device=device,
        )
        return pred.sum()  # 保留计算图

    for i in range(n_coords):
        for j in range(i, n_coords):  # 对称性优化
            if abs(i - j) > threshold:  # 混合方法：限制非对角元素
                continue

            # 生成扰动位置
            pos_pp = data.pos.clone()
            pos_pm = data.pos.clone()
            pos_mp = data.pos.clone()
            pos_mm = data.pos.clone()

            pos_pp.view(-1)[i] += epsilon
            pos_pp.view(-1)[j] += epsilon
            pos_pm.view(-1)[i] += epsilon
            pos_pm.view(-1)[j] -= epsilon
            pos_mp.view(-1)[i] -= epsilon
            pos_mp.view(-1)[j] += epsilon
            pos_mm.view(-1)[i] -= epsilon
            pos_mm.view(-1)[j] -= epsilon


            # 分别计算能量，保留计算图
            energy_pp = energy_func(pos_pp)
            print(f"计算图占用显存：{torch.cuda.memory_allocated() / 1024**2}")
            energy_pm = energy_func(pos_pm)
            print(f"计算图占用显存：{torch.cuda.memory_allocated() / 1024**2}")
            energy_mp = energy_func(pos_mp)
            print(f"计算图占用显存：{torch.cuda.memory_allocated() / 1024**2}")
            energy_mm = energy_func(pos_mm)
            print(f"计算图占用显存：{torch.cuda.memory_allocated() / 1024**2}")

            # 二阶导数公式
            second_derivative = (
                energy_pp - energy_pm - energy_mp + energy_mm
            ) / (4 * epsilon**2)

            hessian[i, j] = second_derivative
            if i != j:
                hessian[j, i] = second_derivative  # 对称性优化
            print(f"结束循环{i}.{j}")

    return hessian


def train_one_epoch_fd(
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

    for step, data in enumerate(data_loader):
        logger.info(f"Processing step {step}/{len(data_loader)}")
        data = data.to(device)
        data.pos.requires_grad_(True)

        with amp_autocast():
            logger.info("Starting Hessian computation using finite differences.")
            batch_hessians = []

            for sample_idx in data.batch.unique():
                sample_mask = data.batch == sample_idx
                sample_pos = data.pos[sample_mask]

                logger.info(f"Sample {sample_idx}: pos.shape = {sample_pos.shape}")
                hessian = compute_hessian_finite_difference(model, data, epsilon=1e-6, device=device)

                # Reshape Hessian
                n = sample_pos.size(0)
                hessian = hessian.view(n, 3, n, 3).permute(0, 2, 1, 3).contiguous()
                hessian = hessian.view(-1, 9)
                batch_hessians.append(hessian)

            hessian_final = torch.cat(batch_hessians, dim=0)

            # 检查维度匹配
            if hessian_final.shape != data.force_constants_all.shape:
                raise ValueError(
                    f"Dimension mismatch: hessian shape {hessian_final.shape}, "
                    f"target shape {data.force_constants_all.shape}"
                )

            data.force_constants_all = data.force_constants_all.to(device)

            # 计算损失
            loss = criterion(hessian_final, data.force_constants_all)

        optimizer.zero_grad()

        if loss_scaler is not None:
            loss_scaler(loss, optimizer, parameters=model.parameters())
        else:
            loss.backward()
            optimizer.step()

        # 清理显存
        del hessian, batch_hessians
        torch.cuda.empty_cache()

        # 更新指标
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
            # pred = model(
            #     batch=data.batch,
            #     f_in=data.x,
            #     edge_src=data.edge_src,
            #     edge_dst=data.edge_dst,
            #     pos=data.pos,
            #     edge_num=data.edge_num,
            #     device=device
            # )
            # pred = pred.view(-1)
            batch_hessians = []
            for sample_idx in data.batch.unique():
                sample_mask = data.batch == sample_idx
                sample_pos = data.pos[sample_mask]
                sample_pos.requires_grad_(True)
                print(f"Sample {sample_idx}: pos.shape = {sample_pos.shape}")
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
                # hessian = torch.autograd.functional.hessian(model_pred, sample_pos, create_graph=True)
                hessian = torch.autograd.functional.hessian(model_pred, sample_pos)

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
            # 一阶梯度
            # grad_outputs = torch.ones_like(pred)
            # grads = torch.autograd.grad(
            #     pred, data.pos, grad_outputs=grad_outputs, create_graph=True, retain_graph=True
            # )[0]

            # 整批 Hessian 计算
            # hessian_matrix = compute_hessian_vectorized(
            #     model=model,
            #     inputs=data.pos,
            #     batch=data.batch,
            #     f_in=data.x,
            #     edge_src=data.edge_src,
            #     edge_dst=data.edge_dst,
            #     edge_num=data.edge_num,
            #     device=device,
            # )  # shape: [N,3,N,3]

        # # 根据 batch 的子图划分 Hessian
        # unique_graphs = data.batch.unique()
        # hessian_list = []
        # for g_id in unique_graphs:
        #     nodes_g = (data.batch == g_id)
        #     # 提取该图子块 (n_g,3,n_g,3)
        #     sub_hessian = hessian_matrix[nodes_g][:, :, nodes_g, :]
        #     n_g = sub_hessian.size(0)
        #     # 重塑成 (n_g², 9)
        #     sub_hessian = sub_hessian.reshape(n_g * n_g, 9)
        #     hessian_list.append(sub_hessian)

        # # 拼接所有子图的 Hessian 得到 (sum of n_g², 9)
        # hessian = torch.cat(hessian_list, dim=0)

        # 计算损失和MAE
        loss = criterion(hessian_final, data.force_constants_all)
        mae = torch.mean(torch.abs(hessian_final - data.force_constants_all))

        loss_metric.update(loss.item(), n=hessian_final.shape[0])
        mae_metric.update(mae.item(), n=hessian_final.shape[0])

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
    # 保存初始模型参数
    before_params = [param.clone().detach() for param in model.parameters()]
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
            # 保存更新后的参数
            after_params = [param.clone().detach() for param in model.parameters()]
            # 比较参数是否发生变化
            for idx, (before, after) in enumerate(zip(before_params, after_params)):
                if not torch.equal(before, after):
                    if logger is not None:
                        logger.info(f"参数已更新: 参数索引 {idx}")
                        param_name = list(model.state_dict().keys())[idx]
                        logger.info(f"更新的参数名称: {param_name}")
                        # logger.info(f"更新前的参数值: {before}")
                        # logger.info(f"更新后的参数值: {after}")
                    break
            else:
                if logger is not None:
                    logger.info("参数未更新")
            before_params = after_params
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


def kfac_hessian_approximation(model_pred_fn, sample_pos, damping=1e-2):
    """
    Compute the Kronecker-Factored Approximation of the Hessian, considering 3D physical interactions.

    Args:
        model_pred_fn: A function that computes the model's prediction given the input positions.
        sample_pos: Tensor containing the positions for the sample.
        damping: A small value added to the diagonal for numerical stability.

    Returns:
        hessian_kfac: A Kronecker-factored approximation of the Hessian.
    """
    n, _ = sample_pos.shape

    # Compute the Jacobian-vector product (VJP)
    def compute_jacobian(vec):
        v = torch.tensor(1.0, device=sample_pos.device)  # Scalar v matching f(pos).sum()
        _, vjp_result = vjp(lambda pos: model_pred_fn(pos).sum(), sample_pos, v)
        return vec * vjp_result

    # Initialize factors for Kronecker-Factored Approximation
    jacobian_vectors = []
    for i in range(n):
        for j in range(3):  # Loop over x, y, z directions
            basis_vector = torch.zeros_like(sample_pos)
            basis_vector[i, j] = 1.0  # Set a single direction to 1
            jacobian_vectors.append(compute_jacobian(basis_vector))

    # Reshape and aggregate Jacobian vectors
    jacobian_vectors = torch.stack(jacobian_vectors, dim=0)  # Shape: [n*3, n, 3]

    # Compute G considering 3D coordinates
    G = torch.einsum('abc,abd->bcd', jacobian_vectors, jacobian_vectors)  # Shape: [n, 3, 3]
    print(G)
    G = torch.mean(G, dim=0)  # Final shape: [3, 3]

    # Compute A for Kronecker factorization
    A = torch.einsum('abc,adc->bd', jacobian_vectors, jacobian_vectors)  # Shape: [n, n]
    # print(A.shape)
    # A = torch.mean(A, dim=0)  # Final shape: [n, n]
    # print(A.shape)

    # Add damping for numerical stability
    A += damping * torch.eye(A.shape[0], device=A.device)  # Ensure A is [n, n]
    # print(A)
    G += damping * torch.eye(G.shape[0], device=G.device)  # Ensure G is [3, 3]


    # Compute Kronecker product approximation
    hessian_kfac = torch.kron(A, G)
    return hessian_kfac


def train_one_epoch_kfac(
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

    for step, data in enumerate(data_loader):
        data = data.to(device)
        data.pos.requires_grad_(True)

        with amp_autocast():
            batch_hessians = []
            for sample_idx in data.batch.unique():
                sample_mask = data.batch == sample_idx
                sample_pos = data.pos[sample_mask]
                sample_pos.requires_grad_(True)

                def model_pred(pos):
                    return model(
                        batch=data.batch[sample_mask],
                        f_in=data.x[sample_mask],
                        edge_src=data.edge_src,
                        edge_dst=data.edge_dst,
                        pos=pos,
                        edge_num=data.edge_num,
                        device=device,
                    )
                logger.info("Starting model forward pass.")
                # Use K-FAC to approximate the Hessian
                hessian_kfac = kfac_hessian_approximation(model_pred, sample_pos)
                print(hessian_kfac)
                batch_hessians.append(hessian_kfac)

            hessian_final = torch.cat(batch_hessians, dim=0)
            hessian_final.requires_grad_(True)

            if hessian_final.shape != data.force_constants_all.shape:
                raise ValueError(
                    f"Dimension mismatch: hessian shape {hessian_final.shape}, "
                    f"target shape {data.force_constants_all.shape}"
                )

            loss = criterion(hessian_final, data.force_constants_all).requires_grad_()

        optimizer.zero_grad()

        if loss_scaler is not None:
            loss_scaler(loss, optimizer, parameters=model.parameters())
        else:
            loss.backward()

            if clip_grad is not None:
                dispatch_clip_grad(model.parameters(), value=clip_grad, mode='norm')

            optimizer.step()

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

def train_one_epoch_adam(
    model: torch.nn.Module, 
    criterion: torch.nn.Module,
    data_loader: Iterable, 
    optimizer: Optimizer,
    device: torch.device, 
    epoch: int, 
    model_ema: Optional['ModelEma'] = None,  # 假设 ModelEma 已定义
    amp_autocast=None,
    loss_scaler=None,
    clip_grad=None,
    print_freq: int = 50, 
    logger=None
    ):
    # 更新参数的速率太少，唯一的用处就是证明了v3模型可以进行反向传播，中间梯度没有断裂
    # 作为近似Hessian的方法，现已弃用
    # 保存初始模型参数
    before_params = [param.clone().detach() for param in model.parameters()]
    model.train()
    criterion.train()

    loss_metric = AverageMeter()
    mae_metric = AverageMeter()

    for step, data in enumerate(data_loader):
        data = data.to(device)
        data.pos.requires_grad_(True)

        # optimizer.zero_grad()

        with amp_autocast():
            batch_hessians = []
            for sample_idx in data.batch.unique():
                sample_mask = data.batch == sample_idx
                sample_pos = data.pos[sample_mask]
                # print(f"sample_idx:{sample_idx};sample_mask:{sample_mask};sample_pos:{sample_pos}")
                n = sample_pos.size(0)
                print(f"当前样本的原子数为：{n}")


                sample_pred = model(
                        batch=data.batch[sample_mask],
                        f_in=data.x[sample_mask],
                        edge_src=data.edge_src,
                        edge_dst=data.edge_dst,
                        pos=sample_pos,
                        edge_num=data.edge_num,
                        device=device,
                    )
                print(f"model_pred的gradfn为：{sample_pred.grad_fn}")
                # print(sample_pred.shape)
                # print(sample_pred)

                # 计算梯度
                grads = torch.autograd.grad(
                    outputs=sample_pred.sum(),
                    inputs=sample_pos,
                    # grad_outputs=torch.ones_like(sample_pred),
                    create_graph=True,
                    retain_graph=True,
                )[0]  # grads 的形状: (n, 3)
                print(f"grads的gradfn为：{grads.grad_fn}")
                # print(grads)

                # 计算梯度的平方并平均
                square_grads = grads.pow(2)  # (n, 3)
                print(f"square_grads的gradfn为：{square_grads.grad_fn}")
                v_t = torch.mean(square_grads, dim=1)  # (n,)
                print(f"v_t的gradfn为：{v_t.grad_fn}")

                # 构造近似 Hessian 矩阵的对角块
                # 使用 v_t 乘以单位矩阵，保持与模型参数的依赖关系
                diag_blocks = v_t.view(n, 1, 1) * torch.eye(3, device=device).view(1, 3, 3)  # (n, 3, 3)
                approx_hessian = diag_blocks.view(n, -1)  # (n, 9)
                print(f"approx_hessian的gradfn为：{approx_hessian.grad_fn}")
                # logger.info(approx_hessian)

                # 构造完整的 Hessian 矩阵，其中对角块为 approx_hessian，非对角块为零
                # 注意：为了保持梯度流动，非对角块不需要参与梯度计算，因此不设置 requires_grad=True
                hessian_blocks = []
                for i in range(n):
                    for j in range(n):
                        if i == j:
                            # 对角块来自 approx_hessian
                            hessian_blocks.append(approx_hessian[i])  # (9,)
                        else:
                            # 非对角块设为零
                            zero_block = torch.zeros(9, device=device, dtype=approx_hessian.dtype)
                            hessian_blocks.append(zero_block)
                # 拼接成 (n^2, 9)
                approx_hessian_full = torch.stack(hessian_blocks, dim=0)  # (n^2, 9)
                print(f"approx_hessian_full的gradfn为：{approx_hessian_full.grad_fn}")
                # logger.info(approx_hessian_full)

                batch_hessians.append(approx_hessian_full)

            # 合并所有样本的 Hessian
            hessian_final = torch.cat(batch_hessians, dim=0)  # (sum n_i^2, 9)
            print(f"hessian_final的gradfn为：{hessian_final.grad_fn}")

            # 确保形状匹配
            if hessian_final.shape != data.force_constants_all.shape:
                raise ValueError(
                    f"Dimension mismatch: hessian shape {hessian_final.shape}, "
                    f"target shape {data.force_constants_all.shape}"
                )

            # 计算损失
            loss = criterion(hessian_final, data.force_constants_all.to(device))
            trace_grad_fn(loss)

        # 反向传播和优化
        if loss_scaler is not None:
            loss_scaler(loss, optimizer, parameters=model.parameters())
        else:
            loss.backward()
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            print(f"loss的grad为{loss.grad}")

        # 保存更新后的参数
        after_params = [param.clone().detach() for param in model.parameters()]
        updated_params = []
        # 比较参数是否发生变化
        for idx, (before, after) in enumerate(zip(before_params, after_params)):
            if not torch.equal(before, after):
                param_name = list(model.state_dict().keys())[idx]
                updated_params.append((idx, param_name, before, after))

                # 随机选择五个更新的参数进行打印
                if updated_params:
                    sample_params = random.sample(updated_params, min(len(updated_params), 5))
                    for idx, param_name, before, after in sample_params:
                        logger.info(f"参数已更新: 参数索引 {idx}")
                        logger.info(f"更新的参数名称: {param_name}")
                        logger.info(f"更新前的参数值: {before}")
                        logger.info(f"更新后的参数值: {after}")
                    # break
                else:
                    logger.info("参数未更新")

                # 更新 before_params
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

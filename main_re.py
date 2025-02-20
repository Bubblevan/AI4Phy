import argparse
import datetime
import itertools
import subprocess
import time

import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
torch.set_default_dtype(torch.float32)

import os
from logger import FileLogger
from pathlib import Path

from datasets.graph import FC
from features.process_data import get_Path,splitdata

# AMP
from contextlib import suppress
from timm.utils import NativeScaler

import nets
from nets import model_entrypoint

from timm.utils import ModelEmaV2
from timm.scheduler import create_scheduler
from engine import *

from nets.loss import FrobeniusNormLoss
# distributed training
import utils
import warnings

import torch.profiler
from torch.profiler import profile, record_function, ProfilerActivity

warnings.filterwarnings('ignore')
ModelEma = ModelEmaV2
num_workers = 12

def get_args_parser():
    parser = argparse.ArgumentParser('Training equivariant networks', add_help=False)
    parser.add_argument('--output-dir', type=str, default=None)
    # network architecture
    parser.add_argument('--model-name', type=str, default='transformer_ti')
    parser.add_argument('--input-irreps', type=str, default=None)
    parser.add_argument('--radius', type=float, default=2.0)
    parser.add_argument('--num-basis', type=int, default=86)
    parser.add_argument('--output-channels', type=int, default=1)
    # training hyper-parameters
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.9999, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')
    # regularization
    parser.add_argument('--drop-path', type=float, default=0.0)
    # optimizer (timm)
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='weight decay (default: 0.01)')
    # learning rate schedule parameters (timm)
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    # logging
    parser.add_argument("--print-freq", type=int, default=50)
    # task
    parser.add_argument("--target", type=int, default=0)
    parser.add_argument("--data-path", type=str, default='datasets')
    parser.add_argument('--run-fold', type=int, default=None)
    parser.add_argument('--order-type', type=str, default='all')
    parser.add_argument('--feature-type', type=str, default='one_hot')
    parser.add_argument('--compute-stats', action='store_true', dest='compute_stats')
    parser.set_defaults(compute_stats=False)
    parser.add_argument('--no-standardize', action='store_false', dest='standardize')
    parser.set_defaults(standardize=True)
    parser.add_argument('--loss', type=str, default='l1')
    # random
    parser.add_argument("--seed", type=int, default=0)
    # data loader config
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',help='')
    parser.set_defaults(pin_mem=True)
    # AMP
    parser.add_argument('--no-amp', action='store_false', dest='amp', 
                        help='Disable FP16 training.')
    parser.set_defaults(amp=True)
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser

# def custom_collate(batch):
#     # Initialize lists to hold the concatenated results
#     force_constants_all_list = []
#     x_list = []
#     pos_list = []
#     edge_src_list = []
#     edge_dst_list = []
#     edge_attr_list = []
#     edge_vec_list = []
#     edge_index_list = []
#     eigenvalues_list = []
#     eigenvectors_list = []

#     total_atoms = 0

#     for data in batch:
#         num_atoms = data.pos.shape[0]
#         print(f"Sample {data} atom count: {num_atoms}")
#         print(f"x.shape: {data.x.shape}, pos.shape: {data.pos.shape}, force_constants_all.shape: {data.force_constants_all.shape}")

#         force_constants_all_list.append(data.force_constants_all)
#         x_list.append(data.x)
#         pos_list.append(data.pos)
#         edge_src_list.append(data.edge_src)
#         edge_dst_list.append(data.edge_dst)
#         edge_attr_list.append(data.edge_attr)
#         edge_vec_list.append(data.edge_vec)
#         edge_index_list.append(data.edge_index + total_atoms)  # Shift indices by total_atoms
#         eigenvalues_list.append(data.eigenvalues)
#         eigenvectors_list.append(data.eigenvectors)

#         total_atoms += num_atoms

#     # Concatenate and check dimensions before doing it
#     try:
#         force_constants_all = torch.cat(force_constants_all_list, dim=0)
#         x = torch.cat(x_list, dim=0)
#         pos = torch.cat(pos_list, dim=0)
#         edge_src = torch.cat(edge_src_list, dim=0)
#         edge_dst = torch.cat(edge_dst_list, dim=0)
#         edge_attr = torch.cat(edge_attr_list, dim=0)
#         edge_vec = torch.cat(edge_vec_list, dim=0)
#         edge_index = torch.cat(edge_index_list, dim=1)
#         eigenvalues = torch.cat(eigenvalues_list, dim=0)
#         eigenvectors = torch.cat(eigenvectors_list, dim=0)
#     except Exception as e:
#         print(f"Error during concatenation: {e}")
#         return None

#     # Return the batch
#     batch_data = Data(x=x, edge_src=edge_src, edge_dst=edge_dst,
#                       pos=pos, force_constants_all=force_constants_all,
#                       edge_attr=edge_attr, edge_vec=edge_vec,
#                       edge_index=edge_index, eigenvalues=eigenvalues,
#                       eigenvectors=eigenvectors)

#     return batch_data


def main(args):

    #utils.init_distributed_mode(args)
    args.distributed = False
    args.rank = 0
    args.local_rank = 0
    # args.distributed = True
    # args.rank = 7
    # args.local_rank = args.rank % torch.cuda.device_count()
    # torch.cuda.set_device(args.local_rank)
    # args.dist_backend = 'nccl'
    # torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
    #                                      world_size=args.world_size, rank=args.rank)
    # torch.distributed.barrier()

    is_main_process = (args.rank == 0)
    _log = FileLogger(is_master=is_main_process, is_rank0=is_main_process, output_dir=args.output_dir)
    _log.info(args)
    root_path = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(root_path+'/best_models/')

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    ''' Dataset '''
    # data_source = get_Path(args.data_path+'/mp/')
    data_source = get_Path(args.data_path+'/testsmall/')


    fold_num = 10
    train_idx,valid_idx,test_idx = splitdata(data_source,fold_num,args.run_fold)

    train = [data_source[i] for i in train_idx]
    valid = [data_source[i] for i in valid_idx]
    test = [data_source[i] for i in test_idx]

    train_dataset = FC(args.data_path,'train', train, args.run_fold)
    val_dataset  = FC(args.data_path, 'valid',valid, args.run_fold)
    test_dataset = FC(args.data_path, 'test',test, args.run_fold)

    ''' Data Loader '''

    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(
                        train_dataset, num_replicas=utils.get_world_size(), rank=utils.get_rank(), shuffle=True
                )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                sampler=sampler_train, num_workers=args.workers, pin_memory=args.pin_mem, 
                drop_last=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                shuffle=True, num_workers=args.workers, pin_memory=args.pin_mem, 
                drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,num_workers=args.workers)
    
    for batch in enumerate(train_loader):
        print(batch)
        # print(f"Loaded batch pos.shape: {batch.pos.shape}")

        break  # 只查看第一个批次
        # for sample_idx in batch.batch.unique():
        #     sample_mask = batch.batch == sample_idx
        #     sample_pos = batch.pos[sample_mask]
        #     print(f"  Sample {sample_idx}: pos.shape = {sample_pos.shape}")
        # break

    # since dataset needs random 
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ##############################################################
    # device = 'cpu' # 看完记得改回去
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    ##############################################################

    # Print data attributes and dimensions
    # for step, data in enumerate(train_loader):
    #     print(f"Batch {step + 1}:")
    #     for attr in dir(data):
    #         if not attr.startswith("_") and hasattr(data, attr):
    #             try:
    #                 value = getattr(data, attr)
    #                 print(f"{attr}: {value.shape if hasattr(value, 'shape') else value}")
    #             except Exception as e:
    #                 print(f"{attr}: Error accessing attribute ({e})")
    #     print("-" * 40)
    #     if step >= 0:  # Print only the first few batches for demonstration
    #         break
    ''' Network '''
    create_model = model_entrypoint(args.model_name)
    model = create_model(irreps_in=args.input_irreps, 
            radius=args.radius, num_basis=args.num_basis, 
            out_channels=args.output_channels, 
            atomref=None, #train_dataset.atomref(args.target),
            drop_path=args.drop_path)
    # _log.info(model)
    model = model.to(device)
    
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
                model,
                decay=args.model_ema_decay,
                device='cpu' if args.model_ema_force_cpu else None)

    # distributed training
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # _log.info('Number of params: {}'.format(n_parameters))
    
    ''' Optimizer and LR Scheduler '''
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = torch.nn.MSELoss()

    ''' AMP (from timm) '''
    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if args.amp:
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
    
    ''' Compute stats '''
    if args.compute_stats:
        compute_stats(train_loader, max_radius=args.radius, logger=_log, print_freq=args.print_freq)
        return

    best_epoch, best_train_err, best_val_err, best_test_err = 0, float('inf'), float('inf'), float('inf')
    best_ema_epoch, best_ema_val_err, best_ema_test_err = 0, 0, float('inf')
    
    for epoch in range(args.epochs):
        print(f"Starting epoch {epoch+1}/{args.epochs}...")
        epoch_start_time = time.perf_counter()
        epoch_error = []
        lr_scheduler.step(epoch)

        train_err, train_loss = train_one_epoch_hessian(
            model=model, criterion=criterion, 
            data_loader=train_loader, optimizer=optimizer,
            device=device, epoch=epoch, model_ema=model_ema, 
            amp_autocast=amp_autocast, loss_scaler=loss_scaler,
            print_freq=args.print_freq, logger=_log)

        val_err, val_loss = evaluate_dx(
            model=model, data_loader=val_loader,
            device=device, amp_autocast=amp_autocast,
            print_freq=args.print_freq, logger=_log)

        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        
        # train_err, train_loss = train_one_epoch_dx(model=model, criterion=criterion, 
        #          data_loader=train_loader, optimizer=optimizer,
        #         device=device, epoch=epoch, model_ema=model_ema, 
        #         amp_autocast=amp_autocast, loss_scaler=loss_scaler,
        #         print_freq=args.print_freq, logger=_log)
        
        # val_err, val_loss = evaluate_dx(model, val_loader, device, 
        #         amp_autocast=amp_autocast, print_freq=args.print_freq, logger=_log)
        
        test_err, test_loss = evaluate_dx(model,  test_loader, device, 
                amp_autocast=amp_autocast, print_freq=args.print_freq, logger=_log)
        
        # record the best results
        if val_loss < best_val_err:
            best_val_err = val_loss
            best_test_err = test_loss
            best_train_err = train_loss
            best_epoch = epoch
            if best_test_err < 100:
                torch.save(model,save_path+str(args.run_fold)+'_save_der.pt')

        # print LOSS
        
        info_str = 'Epoch: [{epoch}] train loss: {train_loss:.5f}, train MAE: {train_mae:.5f},'.format(epoch=epoch,train_loss=train_loss,train_mae=train_err)
        info_str += 'val loss: {val_loss:.5f}, val MAE: {val_mae:.5f},'.format(val_loss=val_loss,val_mae=val_err)
        info_str += 'test loss: {test_loss:.5f}, test MAE: {test_mae:.5f},'.format(test_loss=test_loss,test_mae=test_err)
        info_str += 'Time: {:.2f}s'.format(time.perf_counter() - epoch_start_time)
        _log.info(info_str)
        
        info_str = 'Best -- epoch={}, train loss: {:.5f}, val loss: {:.5f}, test loss: {:.5f}\n'.format(
                best_epoch, best_train_err, best_val_err, best_test_err)
        _log.info(info_str)
        epoch_error.append(best_test_err)
        
        
        # evaluation with EMA
        if model_ema is not None:
            ema_val_err, _ = evaluate(model_ema.module, val_loader, device, 
                    amp_autocast=amp_autocast, print_freq=args.print_freq, logger=_log)
            
            ema_test_err, _ = evaluate(model_ema.module, test_loader, device, 
                    amp_autocast=amp_autocast, print_freq=args.print_freq, logger=_log)
            
            # record the best results
            if (ema_val_err) < best_ema_val_err:
                best_ema_val_err = ema_val_err
                best_ema_test_err = ema_test_err
                best_ema_epoch = epoch

            info_str = 'Epoch: [{epoch}]'.format(epoch=epoch)
            info_str += 'EMA val MAE: {:.5f}, '.format(ema_val_err)
            info_str += 'EMA test MAE: {:.5f}, '.format(ema_test_err)
            info_str += 'Time: {:.2f}s'.format(time.perf_counter() - epoch_start_time)
            _log.info(info_str)
            
            info_str = 'Best EMA -- epoch={}, val MAE: {:.5f}, test MAE: {:.5f}\n'.format(
                        best_ema_epoch, best_ema_val_err, best_ema_test_err)
            _log.info(info_str)
    
    all_err = 'fold_{} test LOSS:{:.5f}'.format(str(args.run_fold),epoch_error[-1])
    _log.info(all_err)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser('Training equivariant networks', parents=[get_args_parser()])
    args = parser.parse_args()      
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
        

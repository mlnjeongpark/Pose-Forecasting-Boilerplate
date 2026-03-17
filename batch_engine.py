import open3d as o3d
import numpy as np
import trimesh
import os
import time
from tqdm import tqdm

import torch
import wandb

from tools.utils import time_str, AverageMeter, save_ckpt
from loss.losses import mse_loss, mpjpe_loss


def train(epoch, train_loader, model, loss_w, optimizer, scheduler, device, save_path):
    model.train()
    running_loss = 0.0
    loss_meter = AverageMeter()
    batch_num = len(train_loader)
    log_interval = 100


    gt_list = []
    pred_list = []

    for step, (obs, pred_gt) in enumerate(train_loader):
        batch_time = time.time()

        obs = obs.to(device)
        pred_gt = pred_gt.to(device)

        pred = model(obs, targets=pred_gt)

        mse = mse_loss(pred, pred_gt)
        mpjpe = mpjpe_loss(pred.view(pred.shape[0], pred.shape[1], 21, 3), 
                           pred_gt.view(pred_gt.shape[0], pred_gt.shape[1], 21, 3))
        loss_list = [mse, mpjpe]
        train_loss = 0
        for i, l in enumerate(loss_w):
            train_loss = train_loss + (loss_list[i] * l)

        optimizer.zero_grad(set_to_none=True)
        train_loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += train_loss.item()

        loss_meter.update(train_loss)

        wandb.log({'epoch': epoch,
                   'step': step,
                   'mse_loss': mse,
                   'mpjpe_loss': mpjpe,
                   'train_loss': train_loss,
                   'lr': optimizer.param_groups[0]['lr']})

        if (step + 1) % log_interval == 0 or (step + 1) == batch_num:
            print(f'{time_str()}, '
                    f'Step {step}/{batch_num} in Ep {epoch}, \n'
                    f'LW: {loss_w} , LR: {optimizer.param_groups[0]["lr"]} , '
                    f'Time: {time.time() - batch_time:.2f}s , \n',
                    f'mse_loss: {mse:.4f}',
                    f'mpjpe_loss: {mpjpe:.4f}, '
                    f'train_loss: {loss_meter.avg:.4f}, \n')
            
        gt_list.append(pred_gt.cpu().numpy())
        pred_list.append(pred.cpu().detach().numpy())

        
    gt_label = np.concatenate(gt_list, axis=0)
    pred_label = np.concatenate(pred_list, axis=0)

    save_ckpt(model, 
            os.path.join(save_path, f'{epoch}_epoch.pth'),
            optimizer,
            scheduler,
            epoch,
            mpjpe) 
    
    return gt_label, pred_label


def eval(epoch, valid_loader, model, loss_w, optimizer, scheduler, device, save_path):
    model.eval()
    running_loss = 0.0
    loss_meter = AverageMeter()
    batch_num = len(valid_loader)
    log_interval = 100


    gt_list = []
    pred_list = []
    with torch.no_grad():
        for step, (obs, pred_gt) in enumerate(tqdm(valid_loader)):
            batch_time = time.time()

            obs = obs.to(device)
            pred_gt = pred_gt.to(device)

            pred = model(obs, targets=pred_gt)

            mse = mse_loss(pred, pred_gt)
            mpjpe = mpjpe_loss(pred.view(pred.shape[0], pred.shape[1], 21, 3), 
                            pred_gt.view(pred_gt.shape[0], pred_gt.shape[1], 21, 3))
            loss_list = [mse, mpjpe]
            valid_loss = 0
            for i, l in enumerate(loss_w):
                valid_loss = valid_loss + (loss_list[i] * l)

            running_loss += valid_loss.item()

            loss_meter.update(valid_loss)

            gt_list.append(pred_gt.cpu().numpy())
            pred_list.append(pred.cpu().detach().numpy())
        
    gt_label = np.concatenate(gt_list, axis=0)
    pred_label = np.concatenate(pred_list, axis=0)
    
    return gt_label, pred_label
import open3d as o3d
import numpy as np
import trimesh
import os
import time
from tqdm import tqdm

import torch
import wandb

from tools.utils import time_str, AverageMeter, save_ckpt
from loss.losses import l1l2_loss, mse_loss, mpjpe_loss


def train(epoch, train_loader, model, loss_w, optimizer, scheduler, device, save_path,model_name):
    model.train()
    model.vp.eval()
    running_loss = 0.0
    loss_meter = AverageMeter()
    batch_num = len(train_loader)
    log_interval = 100


    gt_list = []
    pred_list = []

    for step, (obs, targets) in enumerate(train_loader):
        batch_time = time.time()

        obs = obs.to(device)
        targets = targets.to(device)
        tgt_latent = model._encode_pose_seq(targets)

        pred_pose, pred_latent = model(obs, targets=targets)

        mse = l1l2_loss(pred_latent, tgt_latent)
        mpjpe = mpjpe_loss(pred_pose.view(pred_pose.shape[0], pred_pose.shape[1], 21, 3), 
                           targets.view(targets.shape[0], targets.shape[1], 21, 3))
        # mpjpe = mpjpe_loss(pred, targets)

        loss_list = [mse, mpjpe]
        train_loss = 0
        for i, l in enumerate(loss_w):
            train_loss = train_loss + (loss_list[i] * l)

        optimizer.zero_grad(set_to_none=True)
        train_loss.backward()
        optimizer.step()
        # scheduler.step()

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
            
        gt_list.append(targets.cpu().numpy())
        pred_list.append(pred_pose.cpu().detach().numpy())

        
    gt_label = np.concatenate(gt_list, axis=0)
    pred_label = np.concatenate(pred_list, axis=0)

    save_ckpt(model, 
            os.path.join(save_path, f'{epoch}_epoch.pth'),
            optimizer,
            scheduler,
            epoch,
            mpjpe) 
    
    return gt_label, pred_label


def eval(epoch, valid_loader, model, loss_w, optimizer, scheduler, device, save_path, model_name):
    model.eval()
    running_loss = 0.0
    loss_meter = AverageMeter()
    batch_num = len(valid_loader)
    log_interval = 100


    gt_list = []
    pred_list = []
    with torch.no_grad():
        for step, (obs, targets) in enumerate(tqdm(valid_loader)):
            batch_time = time.time()

            obs = obs.to(device)
            targets = targets.to(device)
            if not (model_name == 'zero' or model_name == 'constant'):
                tgt_latent = model._encode_pose_seq(targets)

            pred_pose = model(obs)

            # mse = l1l2_loss(pred_latent, tgt_latent)
            # mpjpe = mpjpe_loss(pred_pose.view(pred_pose.shape[0], pred_pose.shape[1], 21, 3), 
            #                 targets.view(targets.shape[0], targets.shape[1], 21, 3))
            # loss_list = [mse, mpjpe]
            # valid_loss = 0
            # for i, l in enumerate(loss_w):
            #     valid_loss = valid_loss + (loss_list[i] * l)

            # running_loss += valid_loss.item()

            # loss_meter.update(valid_loss)

            gt_list.append(targets.cpu().numpy())
            pred_list.append(pred_pose.cpu().detach().numpy())
        
    gt_label = np.concatenate(gt_list, axis=0)
    pred_label = np.concatenate(pred_list, axis=0)
    
    return gt_label, pred_label
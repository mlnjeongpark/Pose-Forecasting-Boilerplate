"""
CUDA_VISIBLE_DEVICES=0 nohup python train.py --obs 15 --pred 30 --model mlp --layer 3 --dim 128 > mlp.log &
CUDA_VISIBLE_DEVICES=0 nohup python train.py --obs 30 --pred 30 --model mlp --layer 3 --dim 128 > mlp.log &
CUDA_VISIBLE_DEVICES=0 nohup python train.py --obs 30 --pred 60 --model mlp --layer 3 --dim 128 > mlp.log &
CUDA_VISIBLE_DEVICES=0 nohup python train.py --obs 30 --pred 90 --model mlp --layer 3 --dim 128 > mlp.log &
CUDA_VISIBLE_DEVICES=0 nohup python train.py --obs 60 --pred 90 --model mlp --layer 3 --dim 128 > mlp.log & 
CUDA_VISIBLE_DEVICES=1 nohup python train.py --obs 15 --pred 30 --model rnn --layer 1 --dim 256 > rnn.log & 
CUDA_VISIBLE_DEVICES=7 nohup python train.py --obs 30 --pred 30 --model rnn --layer 1 --dim 256 > rnn.log & 
CUDA_VISIBLE_DEVICES=8 nohup python train.py --obs 30 --pred 60 --model rnn --layer 1 --dim 256 > rnn.log & 
CUDA_VISIBLE_DEVICES=9 nohup python train.py --obs 30 --pred 90 --model rnn --layer 1 --dim 256 > rnn.log & 
CUDA_VISIBLE_DEVICES=1 nohup python train.py --obs 60 --pred 90 --model rnn --layer 1 --dim 256 > rnn.log & 
CUDA_VISIBLE_DEVICES=2 nohup python train.py --obs 15 --pred 30 --model transformer > transformer.log & 
CUDA_VISIBLE_DEVICES=3 nohup python train.py --obs 30 --pred 30 --model transformer > transformer.log & 
CUDA_VISIBLE_DEVICES=4 nohup python train.py --obs 30 --pred 60 --model transformer > transformer.log & 
CUDA_VISIBLE_DEVICES=5 nohup python train.py --obs 30 --pred 90 --model transformer > transformer.log & 
CUDA_VISIBLE_DEVICES=6 nohup python train.py --obs 60 --pred 90 --model transformer > transformer.log & 

"""

import open3d as o3d
import numpy as np
import trimesh
import os
import time
import argparse
import wandb

import torch
from torch.utils.data import DataLoader
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser

from configs import cfg, update_config
from dataset.dataloader import PoseDataset
from model.transformer import PoseTransformer, PoseTransformerConfig
from model.baseline import ZeroVelocity, ConstantVelocity
from model.mlp import PoseMLP, PoseMLPConfig
from model.rnn import PoseRNN, PoseRNNConfig
from model.dit import DiffusionTransformer, DiffusionTransformerConfig
from tools.utils import time_str, AverageMeter, save_ckpt, set_seed
from metric.metric import evaluate_metrics, mpjpe_at_intervals
from batch_engine import train, eval

set_seed(123)

def main(cfg, args):
    cur_time = time_str()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device is', device)
    # if torch.cuda.is_available():
    #     os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    model_name = cfg.MODEL.NAME
    
    wandb.init(project=f'Pose-Forecasting_{cfg.MODEL.NAME}_obs{cfg.DATA.OBS}_pred{cfg.DATA.PRED}'
                # ,mode="disabled"
            )

    wandb.run.name = f'layer{cfg.TRANSFORMER.LAYER}-dim{cfg.TRANSFORMER.EMBED}-lr-{cfg.TRAIN.LR}_wd-{cfg.TRAIN.WD}-{cur_time}'

    total_epoch = cfg.TRAIN.EPOCH
    save_path = os.path.join('saved_model',f'{cfg.MODEL.NAME}', f'{cfg.DATA.OBS}_{cfg.DATA.PRED}' ,time_str())
    if not os.path.exists(os.path.abspath(save_path)):
        os.makedirs(os.path.abspath(save_path))

    print(cfg)

    # Load Data
    train_ds = PoseDataset(root='data', 
                        split='train', 
                        device=device,
                        obs_len=cfg.DATA.OBS,
                        pred_len=cfg.DATA.PRED,
                        stride=cfg.DATA.STRIDE,)

    test_ds = PoseDataset(root='data', 
                        split='test', 
                        device=device,
                        obs_len=cfg.DATA.OBS,
                        pred_len=cfg.DATA.PRED,
                        stride=cfg.DATA.STRIDE,)

    train_loader = DataLoader(train_ds, batch_size=cfg.TRAIN.BATCH, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.TRAIN.BATCH, shuffle=False, num_workers=4, pin_memory=True)

    # Load Model
    bm = BodyModel(bm_fname='data/VPoserModelFiles/smplx_neutral_model.npz').to(device)
    vp, ps = load_model('data/VPoserModelFiles/vposer_v2_05', model_code=VPoser,
                                remove_words_in_model_weights='vp_model.',
                                disable_grad=True,
                                comp_device=device)
    vp = vp.to(device)

    if model_name == 'transformer':
        config = PoseTransformerConfig(
        obs_len=cfg.DATA.OBS,
        pred_len=cfg.DATA.PRED,
        pose_dim=63,
        latent_dim=32,
        n_layer=cfg.TRANSFORMER.LAYER,
        n_head=cfg.TRANSFORMER.HEAD,
        n_embd=cfg.TRANSFORMER.EMBED,
        dropout=cfg.TRANSFORMER.DROPOUT,
        )
        model = PoseTransformer(vp, config).to(device)

    elif model_name == 'zero':
        model = ZeroVelocity(vp, pred_len=cfg.DATA.PRED).to(device)
    
    elif model_name == 'constant':
        model = ConstantVelocity(vp, pred_len=cfg.DATA.PRED).to(device)
    
    elif model_name == 'mlp':
        config = PoseMLPConfig(
        obs_len=cfg.DATA.OBS,
        pred_len=cfg.DATA.PRED,
        pose_dim=63,
        latent_dim=32,
        n_layer=cfg.TRANSFORMER.LAYER,
        hidden_dim=cfg.TRANSFORMER.EMBED,
        dropout=cfg.TRANSFORMER.DROPOUT,
        )
        model = PoseMLP(vp, config).to(device)
    
    elif model_name == 'rnn':
        config = PoseRNNConfig(
        obs_len=cfg.DATA.OBS,
        pred_len=cfg.DATA.PRED,
        pose_dim=63,
        latent_dim=32,
        hidden_dim=cfg.TRANSFORMER.EMBED,
        n_layer=cfg.TRANSFORMER.LAYER,
        )
        model = PoseRNN(vp, config).to(device)

    elif model_name == 'diff':
        config = DiffusionTransformerConfig(
            obs_len=cfg.DATA.OBS,
            pred_len=cfg.DATA.PRED,
            pose_dim=63,
            latent_dim=32,
            n_layer=cfg.TRANSFORMER.LAYER,
            n_embd=cfg.TRANSFORMER.EMBED,
            dropout=cfg.TRANSFORMER.DROPOUT,
            diffusion_steps=100,
        )

        model = DiffusionTransformer(vp, config).to(device)

    
    else:
        return

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WD)
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                           T_max=total_epoch,
                                                        #    T_max=total_epoch * len(train_loader),
                                                           eta_min=1e-6)

    # Loss
    loss_w = cfg.TRAIN.LOSS_W

    # Train
    for e in range(total_epoch):
        if (model_name == 'zero' or model_name == 'constant'):
            test_gt, test_pred = eval(e, test_loader, model, loss_w, optimizer, scheduler, device, save_path,model_name)
            test_mpjpe, test_ade, test_fde = evaluate_metrics(test_pred, test_gt)
            test_mpjpe_itv = mpjpe_at_intervals(test_pred, test_gt)
            print("\n==> Test at different horizons")
            for k,v in test_mpjpe_itv.items():
                print(f"{k} ms : {v:.3f}")
                wandb.log({'epoch': e,
                    f'Test {k} ms': v
                    })
            print("-"*80)
            print(f"ADE -  Test: {test_ade}")
            wandb.log({'epoch': e,
                'Test ADE': test_ade, 
                })
            print("-"*80)
            print(f"FDE -  Test: {test_fde}")
            wandb.log({'epoch': e,
                'Test FDE': test_fde, 
                })
            return
        train_gt, train_pred = train(e, train_loader, model, loss_w, optimizer, scheduler, device, save_path,model_name)
        scheduler.step()
        test_gt, test_pred = eval(e, test_loader, model, loss_w, optimizer, scheduler, device, save_path,model_name)
        train_mpjpe, train_ade, train_fde = evaluate_metrics(train_pred, train_gt)
        test_mpjpe, test_ade, test_fde = evaluate_metrics(test_pred, test_gt)
        train_mpjpe_itv = mpjpe_at_intervals(train_pred, train_gt)
        test_mpjpe_itv = mpjpe_at_intervals(test_pred, test_gt)

        print("="*80)
        print(f" Epoch {e} Result")
        print("-"*80)
        print(f"MPJPE - Train: {train_mpjpe}, Test: {test_mpjpe}")
        print("\n==> Train at different horizons")
        for k,v in train_mpjpe_itv.items():
            print(f"{k} ms : {v:.3f}")
            wandb.log({'epoch': e,
                        f'Train {k} ms': v
                        })
        print("\n==> Test at different horizons")
        for k,v in test_mpjpe_itv.items():
            print(f"{k} ms : {v:.3f}")
            wandb.log({'epoch': e,
                f'Test {k} ms': v
                })
        
        print("-"*80)
        print(f"ADE - Train: {train_ade}, Test: {test_ade}")
        wandb.log({'epoch': e,
            'Train ADE': train_ade, 'Test ADE': test_ade, 
            })
        print("-"*80)
        print(f"FDE - Train: {train_fde}, Test: {test_fde}")
        wandb.log({'epoch': e,
            'Train FDE': train_fde, 'Test FDE': test_fde, 
            })
        print("="*80)



def argument_parser():
    parser = argparse.ArgumentParser(description="Transformer based Pose Forecasting",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
    "--cfg", help="decide which cfg to use", type=str,
    default="./configs/pose.yaml")
    
    parser.add_argument("--lr", type=float,default=None)
    parser.add_argument("--dim", type=int,default=None)
    parser.add_argument("--wd", type=float,default=None)
    parser.add_argument("--obs", type=int,default=None)
    parser.add_argument("--pred", type=int,default=None)
    parser.add_argument("--layer", type=int,default=None)
    parser.add_argument("--model", type=str,default=None)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argument_parser()

    update_config(cfg, args)
    main(cfg, args)
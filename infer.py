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
from tools.utils import time_str, AverageMeter, save_ckpt, set_seed, get_reload_weight
from metric.metric import evaluate_metrics, mpjpe_at_intervals
from batch_engine import train, eval

from tqdm import tqdm
from tools.utils import AverageMeter


def main(cfg, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device is', device)

    test_ds = PoseDataset(root='data', 
                        split='test', 
                        device=device,
                        obs_len=cfg.DATA.OBS,
                        pred_len=cfg.DATA.PRED,
                        stride=cfg.DATA.STRIDE,)

    test_loader = DataLoader(test_ds, batch_size=cfg.TRAIN.BATCH, shuffle=False, num_workers=4, pin_memory=True)

    bm = BodyModel(bm_fname='data/VPoserModelFiles/smplx_neutral_model.npz').to(device)
    vp, ps = load_model('data/VPoserModelFiles/vposer_v2_05', model_code=VPoser,
                                remove_words_in_model_weights='vp_model.',
                                disable_grad=True,
                                comp_device=device)
    vp = vp.to(device)

    model_name = cfg.MODEL.NAME

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
            
            gt_list.append(targets.cpu().numpy())
            pred_list.append(pred_pose.cpu().detach().numpy())
        
    gt_label = np.concatenate(gt_list, axis=0)
    pred_label = np.concatenate(pred_list, axis=0)

    test_mpjpe, test_ade, test_fde = evaluate_metrics(pred_label, gt_label)
    test__mpjpe_itv = mpjpe_at_intervals(pred_label, gt_label)

    print("\n==> Test at different horizons")
    for k,v in test__mpjpe_itv.items():
        print(f"{k} ms : {v:.3f}")
    print("-"*80)
    print(f"ADE - Test: {test_ade}")
    print("-"*80)
    print(f"FDE - Test: {test_fde}")
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
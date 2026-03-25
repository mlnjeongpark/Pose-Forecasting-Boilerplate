import open3d as o3d
import numpy as np
import os
import time
import argparse
import wandb
import trimesh


import torch
from torch.utils.data import DataLoader
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.model_loader import load_model
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.models.vposer_model import VPoser

from configs import cfg, update_config
from dataset.dataloader import PoseDataset
from model.transformer import PoseTransformer, PoseTransformerConfig
from tools.utils import time_str, AverageMeter, save_ckpt, set_seed, get_reload_weight
from metric.metric import evaluate_metrics, mpjpe_at_intervals
from batch_engine import train, eval

from tqdm import tqdm
from tools.utils import AverageMeter

from model.baseline import ZeroVelocity, ConstantVelocity
from model.mlp import PoseMLP, PoseMLPConfig
from model.rnn import PoseRNN, PoseRNNConfig
from model.dit import DiffusionTransformer, DiffusionTransformerConfig
def main(cfg, args):
    if args.model == 'mlp':
        ckpt_dir = 'saved_model/2026-03-22_17:27:47/140_epoch.pth'
    if args.model == 'rnn':
        ckpt_dir = 'saved_model/rnn/15_30/2026-03-22_19:36:11/73_epoch.pth'
    if args.model == 'transformer':
        ckpt_dir = 'saved_model/2026-03-19_18:45:51/499_epoch.pth'


    save_dir = f'vis/{args.model}'
    # os.makedirs(f'vis/{cur_time}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

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
    model.eval()
    model = get_reload_weight(
                            model_pth=ckpt_dir,
                            model=model)

    gt_list = []
    pred_list = []
    with torch.no_grad():
        for step, (obs, targets) in enumerate(tqdm(test_loader)):
            batch_time = time.time()

            obs = obs.to(device)
            targets = targets.to(device)

            pred_pose = model(obs)

            gt_list.append(targets.cpu().numpy())
            pred_list.append(pred_pose.cpu().detach().numpy())
        
    gt_label = np.concatenate(gt_list, axis=0)
    pred_label = np.concatenate(pred_list, axis=0)

    # reshape
    pred_ = pred_label.reshape(-1, cfg.DATA.PRED, 21, 3)
    gt_   = gt_label.reshape(-1, cfg.DATA.PRED, 21, 3)
    
    prev_pred = None
    for bch in range(gt_label.shape[0]):
        for fms in range(gt_label.shape[1]):
            best_gt = gt_label[bch, fms]
            best_pred = pred_label[bch, fms]
            
            originalPoses = {'pose_body':torch.from_numpy(best_gt).unsqueeze(0).cuda()}
            recoveredPoses = {'pose_body':torch.from_numpy(best_pred).unsqueeze(0).cuda()}

            bmodelorig = bm(**originalPoses) #bm = BodyModel(bm_fname=bm_fname).to(device) -> SMPL body model인거고 pose parameter를 넣으면 3D 사람 메쉬를 만들어주는 함수
            bmodelreco = bm(**recoveredPoses)
            vorig = c2c(bmodelorig.v) # original # .v는 mesh의 vertex 좌표
            vreco = c2c(bmodelreco.v) # recovered
            faces = c2c(bm.f) #

            mesh1 = trimesh.base.Trimesh(vorig.squeeze(0), faces)
            mesh1.visual.vertex_colors = [254, 254, 254]
            mesh2 = trimesh.base.Trimesh(vreco.squeeze(0), faces)
            mesh2.visual.vertex_colors = [254, 66, 200]
            mesh2.apply_translation([1, 0, 0])  #use [0,0,0] to overlay them on each other
            scene = trimesh.Scene([mesh1, mesh2])
            scene.export(f"{save_dir}/{bch}_{fms}.glb")

            
            mesh1 = trimesh.base.Trimesh(vorig.squeeze(0), faces)
            mesh1.visual.vertex_colors = [254, 254, 254]
            mesh2 = trimesh.base.Trimesh(vreco.squeeze(0), faces)
            mesh2.visual.vertex_colors = [254, 66, 200]
            mesh2.apply_translation([0, 0, 0]) 
            scene = trimesh.Scene([mesh1, mesh2])
            scene.export(f"{save_dir}/{bch}_{fms}_overlay.glb")
            meshes = [mesh1, mesh2]
            joints = c2c(bmodelorig.Jtr).squeeze(0)
            origjoints = joints[0:23, :]   #ignore finger joints
            joints = c2c(bmodelreco.Jtr).squeeze(0) 
            recojoints = joints[0:23, :]  #ignore finger joints

            # print(origjoints.shape, recojoints.shape)
            for i in range(origjoints.shape[0]):
                sphere = trimesh.primitives.Sphere(radius=.02, center=origjoints[i,:])
                sphere.apply_translation([1, 0, 0])
                sphere.visual.vertex_colors = [254, 254, 254]
                meshes.append(sphere)
                sphere = trimesh.primitives.Sphere(radius=.02, center=recojoints[i,:])
                sphere.apply_translation([1, 0, 0])
                sphere.visual.vertex_colors = [254, 150, 200]
                meshes.append(sphere)

            scene2 = trimesh.Scene(meshes)
            # scene2.export(f"vis/{cur_time}/sphere_{bch}_{fms}.glb")
            scene2.export(f"{save_dir}/{bch}_{fms}_sphere.glb")

            if prev_pred is None:
                prev_pred = best_pred
            else:
                prevPoses = {'pose_body':torch.from_numpy(prev_pred).unsqueeze(0).cuda()}
                bmodelprev = bm(**prevPoses)
                preco = c2c(bmodelprev.v) # recovered
                mesh3 = trimesh.base.Trimesh(preco.squeeze(0), faces)
                mesh3.visual.vertex_colors = [0, 66, 200]
                mesh3.apply_translation([1, 0, 0])  #use [0,0,0] to overlay them on each other
                scene = trimesh.Scene([mesh2, mesh3])
                scene.export(f"{save_dir}/{bch}_{fms}_overlap.glb")

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
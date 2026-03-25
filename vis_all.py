import open3d as o3d
import numpy as np
import os
import time
import argparse
import wandb
import trimesh

os.environ["PYOPENGL_PLATFORM"] = "egl"  
# os.environ["EGL_DEVICE_ID"] = "0"       

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
import pyrender
from PIL import Image, ImageDraw, ImageFont

def render_scene_to_image(mesh_list, resolution=(1200, 600)):
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0])

    # 전체 bounding box 중심 계산
    mins = []
    maxs = []
    for m in mesh_list:
        b = m.bounds
        mins.append(b[0])
        maxs.append(b[1])

    scene_min = np.min(np.array(mins), axis=0)
    scene_max = np.max(np.array(maxs), axis=0)
    scene_center = (scene_min + scene_max) / 2.0

    # 중심을 원점 쪽으로 맞춤
    centered_meshes = []
    for m in mesh_list:
        m2 = m.copy()
        m2.apply_translation(-scene_center)
        centered_meshes.append(m2)

    for m in centered_meshes:
        pm = pyrender.Mesh.from_trimesh(m, smooth=False)
        scene.add(pm)

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.5)
    camera_pose = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, -0.1],
        [0, 0, 1, 4.2],
        [0, 0, 0, 1],
    ])
    scene.add(camera, pose=camera_pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=camera_pose)

    r = pyrender.OffscreenRenderer(resolution[0], resolution[1])
    color, _ = r.render(scene)
    r.delete()

    return color
def main(cfg, args):
    mlp_ckpt_dir = 'saved_model/2026-03-22_17:27:47/140_epoch.pth'
    rnn_ckpt_dir = 'saved_model/rnn/15_30/2026-03-22_19:36:11/73_epoch.pth'
    trans_ckpt_dir = 'saved_model/2026-03-19_18:45:51/499_epoch.pth'


    save_dir = f'vis/all'
    # os.makedirs(f'vis/{cur_time}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(os.path.join(save_dir,'jpg'))

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
    transformer = PoseTransformer(vp, config).to(device)

    transformer.eval()
    transformer = get_reload_weight(
                            model_pth=trans_ckpt_dir,
                            model=transformer)


    zero = ZeroVelocity(vp, pred_len=cfg.DATA.PRED).to(device)
    zero.eval()

    
    constant = ConstantVelocity(vp, pred_len=cfg.DATA.PRED).to(device)
    constant.eval()
    
    config = PoseMLPConfig(
        obs_len=cfg.DATA.OBS,
        pred_len=cfg.DATA.PRED,
        pose_dim=63,
        latent_dim=32,
        n_layer=cfg.TRANSFORMER.LAYER,
        hidden_dim=cfg.TRANSFORMER.EMBED,
        dropout=cfg.TRANSFORMER.DROPOUT,
        )
    mlp = PoseMLP(vp, config).to(device)

    mlp.eval()
    mlp = get_reload_weight(
                            model_pth=mlp_ckpt_dir,
                            model=mlp)

    
    config = PoseRNNConfig(
        obs_len=cfg.DATA.OBS,
        pred_len=cfg.DATA.PRED,
        pose_dim=63,
        latent_dim=32,
        hidden_dim=cfg.TRANSFORMER.EMBED,
        )
    rnn = PoseRNN(vp, config).to(device)

    rnn.eval()
    rnn = get_reload_weight(
                            model_pth=rnn_ckpt_dir,
                            model=rnn)


    gt_list = []
    pred_list_zero = []
    pred_list_constant = []
    pred_list_mlp = []
    pred_list_rnn = []
    pred_list_trans = []

    with torch.no_grad():
        for step, (obs, targets) in enumerate(tqdm(test_loader)):
            batch_time = time.time()

            obs = obs.to(device)
            targets = targets.to(device)

            pred_pose_zero = zero(obs)
            pred_pose_constant = constant(obs)
            pred_pose_mlp = mlp(obs)
            pred_pose_rnn = rnn(obs)
            pred_pose_trans = transformer(obs)


            gt_list.append(targets.cpu().numpy())
            pred_list_zero.append(pred_pose_zero.cpu().detach().numpy())
            pred_list_constant.append(pred_pose_constant.cpu().detach().numpy())
            pred_list_mlp.append(pred_pose_mlp.cpu().detach().numpy())
            pred_list_rnn.append(pred_pose_rnn.cpu().detach().numpy())
            pred_list_trans.append(pred_pose_trans.cpu().detach().numpy())

    gt_label = np.concatenate(gt_list, axis=0)
    pred_label_zero = np.concatenate(pred_list_zero, axis=0)
    pred_label_constant = np.concatenate(pred_list_constant, axis=0)
    pred_label_mlp = np.concatenate(pred_list_mlp, axis=0)
    pred_label_rnn = np.concatenate(pred_list_rnn, axis=0)
    pred_label_trans = np.concatenate(pred_list_trans, axis=0)

    # reshape

    prev_pred = None
    for bch in tqdm(range(gt_label.shape[0])):
        for fms in range(gt_label.shape[1]):
            best_gt = gt_label[bch, fms]
            best_predz = pred_label_zero[bch, fms]
            best_predc = pred_label_constant[bch, fms]
            best_predm = pred_label_mlp[bch, fms]
            best_predr = pred_label_rnn[bch, fms]
            best_predt = pred_label_trans[bch, fms]
            
            originalPoses = {'pose_body':torch.from_numpy(best_gt).unsqueeze(0).cuda()}
            recoveredPosesz = {'pose_body':torch.from_numpy(best_predz).unsqueeze(0).cuda()}
            recoveredPosesc = {'pose_body':torch.from_numpy(best_predc).unsqueeze(0).cuda()}
            recoveredPosesm = {'pose_body':torch.from_numpy(best_predm).unsqueeze(0).cuda()}
            recoveredPosesr = {'pose_body':torch.from_numpy(best_predr).unsqueeze(0).cuda()}
            recoveredPosest = {'pose_body':torch.from_numpy(best_predt).unsqueeze(0).cuda()}

            bmodelorig = bm(**originalPoses) #bm = BodyModel(bm_fname=bm_fname).to(device) -> SMPL body model인거고 pose parameter를 넣으면 3D 사람 메쉬를 만들어주는 함수
            bmodelrecoz = bm(**recoveredPosesz)
            bmodelrecoc = bm(**recoveredPosesc)
            bmodelrecom = bm(**recoveredPosesm)
            bmodelrecor = bm(**recoveredPosesr)
            bmodelrecot = bm(**recoveredPosest)
            vorig = c2c(bmodelorig.v) # original # .v는 mesh의 vertex 좌표
            vrecoz = c2c(bmodelrecoz.v) # recovered
            vrecoc = c2c(bmodelrecoc.v) # recovered
            vrecom = c2c(bmodelrecom.v) # recovered
            vrecor = c2c(bmodelrecor.v) # recovered
            vrecot = c2c(bmodelrecot.v) # recovered
            faces = c2c(bm.f) #

            # GT
            mesh_list = []

            mesh_gt = trimesh.Trimesh(vorig.squeeze(0), faces)
            mesh_gt.visual.vertex_colors = [254, 254, 254]
            mesh_list.append(mesh_gt)

            # preds + 색 + 위치
            preds = [
                (vrecoz, [255, 0, 0]),      # red
                (vrecoc, [0, 255, 0]),      # green
                (vrecom, [0, 0, 255]),      # blue
                (vrecor, [255, 255, 0]),    # yellow
                (vrecot, [255, 0, 255])     # magenta
            ]

            for i, (v, color) in enumerate(preds):
                mesh = trimesh.Trimesh(v.squeeze(0), faces)
                mesh.visual.vertex_colors = color
                
                # 옆으로 배치 (x축 이동)
                mesh.apply_translation([i + 1, 0, 0])
                
                mesh_list.append(mesh)

            # scene 생성
            scene = trimesh.Scene(mesh_list)
            scene.export(f"{save_dir}/{bch}_{fms}.glb")

            img = render_scene_to_image(mesh_list)
            pil_img = Image.fromarray(img)

            draw = ImageDraw.Draw(pil_img)

            # 폰트 (없으면 기본 폰트 사용)
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except:
                font = ImageFont.load_default()

            # mesh 개수 (GT + preds)
            num_mesh = len(mesh_list)

            # 이미지 크기
            w, h = pil_img.size

            # 각 mesh 위에 텍스트 위치 계산
            for i in range(num_mesh):
                x = int((i + 0.5) * w / num_mesh)  # 가운데 정렬
                y = 30  # 위쪽에 고정

                if i == 0:
                    text = "GT"
                else:
                    labels = ["Zero", "Const", "MLP", "RNN", "Trans"]
                    text = labels[i - 1]

                draw.text((x, y), text, fill=(255, 255, 255), font=font, anchor="mm")

            pil_img.save(f"{save_dir}/jpg/{bch}_{fms}.jpg")



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
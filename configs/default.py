from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN

_C = CN()


# ----- BASIC SETTINGS -----
_C.NAME = "pose"

_C.TRAIN = CN()
_C.TRAIN.EPOCH = 100
_C.TRAIN.BATCH = 1
_C.TRAIN.LR = 1e-4
_C.TRAIN.WD = 1e-2
_C.TRAIN.LOSS_W = [1, 1]

_C.DATA = CN()
_C.DATA.OBS = 30
_C.DATA.PRED = 30
_C.DATA.STRIDE = 1

_C.MODEL = CN()
_C.MODEL.NAME = 'transformer'


_C.TRANSFORMER = CN()
_C.TRANSFORMER.LAYER = 4
_C.TRANSFORMER.HEAD = 4
_C.TRANSFORMER.EMBED = 128
_C.TRANSFORMER.DROPOUT = 0.1

_C.CUDA = CN()
_C.CUDA.N_GPU = '0'


def update_config(cfg, args):
    cfg.defrost()

    cfg.merge_from_file(args.cfg)  # update cfg
    # cfg.merge_from_list(args.opts)
    if args.lr is not None:
        cfg.TRAIN.LR = args.lr

    if args.wd is not None:
        cfg.TRAIN.WD = args.wd

    if args.obs is not None:
        cfg.DATA.OBS = args.obs

    if args.pred is not None:
        cfg.DATA.PRED = args.pred

    if args.layer is not None:
        cfg.TRANSFORMER.LAYER = args.layer

    if args.dim is not None:
        cfg.TRANSFORMER.EMBED = args.dim
    
    if args.model is not None:
        cfg.MODEL.NAME = args.model
        
    cfg.freeze()
import datetime
import torch
import os
import random
import numpy as np
from collections import OrderedDict

def time_str(fmt=None):
    if fmt is None:
        fmt = '%Y-%m-%d_%H:%M:%S'

    #     time.strftime(format[, t])
    return datetime.datetime.today().strftime(fmt)

class AverageMeter(object):
    """ 
    Computes and stores the average and current value

    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (self.count + 1e-20)


def save_ckpt(model, ckpt_files, optimizer, scheduler, epoch, metric):

    if not os.path.exists(os.path.dirname(os.path.abspath(ckpt_files))):
        os.makedirs(os.path.dirname(os.path.abspath(ckpt_files)))

    save_dict = {'state_dicts': model.state_dict(),
                 'optimizer': optimizer,
                 'scheduler': scheduler,
                 'epoch': f'{time_str()} in epoch {epoch}',
                 'metric': metric,}

    torch.save(save_dict, ckpt_files)

def set_seed(rand_seed):
    # np.random.seed(rand_seed)
    # random.seed(rand_seed)
    # torch.backends.cudnn.enabled = True
    # torch.manual_seed(rand_seed)
    # torch.cuda.manual_seed(rand_seed)
    
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)  


def get_reload_weight(model_pth, model):
    # model_path = os.path.join(model_pth, pth)
    load_dict = torch.load(model_pth, weights_only=False, map_location=lambda storage, loc: storage)

    if isinstance(load_dict, OrderedDict):
        pretrain_dict = load_dict
    else:
        pretrain_dict = load_dict['state_dicts']
        print(f"best performance {load_dict['metric']} in epoch : {load_dict['epoch']}")
    
    model.load_state_dict(pretrain_dict, strict=True)

    return model
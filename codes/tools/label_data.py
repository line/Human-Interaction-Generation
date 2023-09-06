import os
import argparse
import numpy as np
from os.path import join as pjoin
from os.path import dirname, abspath
import json
import sys
sys.path.append(dirname(dirname(abspath(__file__))))

import utils.paramUtil as paramUtil
from options.train_options import TrainCompOptions
from utils.plot_script import *
from utils.get_opt import get_opt

from models import MotionInteractionTransformer
from trainers import DDPMMulTrainer
from datasets import Text2MotionMulDataset

from mmcv.runner import get_dist_info, init_dist
from mmcv.parallel import MMDistributedDataParallel
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from data.NTURGBD_multi.language_labels import ntu_action_multi_enumerator

caps = []
paired_indices = []
tmp = 0
for key in ntu_action_multi_enumerator:
    caps.append(ntu_action_multi_enumerator[key])
    paired_indices.append(list(range(tmp, tmp+len(ntu_action_multi_enumerator[key]))))
    tmp+=len(ntu_action_multi_enumerator[key])


def build_models(opt, dim_pose):
    encoder = MotionInteractionTransformer(
        input_feats=dim_pose,
        num_frames=opt.max_motion_length,
        num_layers=opt.num_layers,
        latent_dim=opt.latent_dim,
        no_clip=opt.no_clip,
        no_eff=opt.no_eff,
        cap_id=opt.cap_id
        )
    return encoder

def setup(rank, world_size, opt):
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def ddp_train(local_rank, world_size, opt):
    setup(local_rank, world_size, opt)

    opt.device = torch.device(local_rank)
    torch.autograd.set_detect_anomaly(True)

    if local_rank == 0:
        os.makedirs(opt.model_dir, exist_ok=True)
        os.makedirs(opt.meta_dir, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    dim_pose = 263
    mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
    std = np.load(pjoin(opt.data_root, 'Std.npy'))

    encoder = build_models(opt, dim_pose)

    checkpoint = torch.load(pjoin(opt.model_dir, opt.which_epoch + '.tar'))
    encoder.load_my_state_dict(checkpoint['encoder'], opt)
    if world_size > 1:
        encoder = MMDistributedDataParallel(
            encoder.cuda(local_rank),
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=False) ## check if unesed parameter exists
    else:
        encoder = encoder.cuda()
    if opt.save_label:
        with open(pjoin(opt.save_root, 'pit_labels.json')) as f:
            model_roles = json.load(f)
        learned_indices = []
        tmp = 0
        for cat_id in range(len(paired_indices)):
            one_role = model_roles[str(cat_id)]
            if 'active_index' in one_role:
                # symmetric interactions
                learned_indices.extend([one_role['active_index'], one_role['passive_index']])
                tmp+=2
            else:
                # asymmetric interactions
                learned_indices.append(tmp)
                tmp+=1
        train_split_file = pjoin(opt.data_root, 'train_sub.txt')
        trainer = DDPMMulTrainer(opt, encoder)
        train_dataset = Text2MotionMulDataset(opt, mean.copy(), std.copy(), train_split_file, opt.times)
        trainer.eval_data(train_dataset, local_rank, world_size, learned_indices, save_dir='data/NTURGBD_multi/pseudo_labels')
    elif opt.label_model:
        train_split_file = pjoin(opt.data_root, 'test_ann_ids.txt')
        trainer = DDPMMulTrainer(opt, encoder)
        train_dataset = Text2MotionMulDataset(opt, mean.copy(), std.copy(), train_split_file, opt.times, label_path=opt.label_path)
        learned_indices = trainer.eval_data(train_dataset, local_rank, world_size, None)
        model_roles = {}
        tmp = 0
        for cat_id, cat_indices in enumerate(paired_indices):
            if len(cat_indices)==1:
                # symmetric interactions
                model_roles[cat_id] = {'category':caps[cat_id]}
            else:
                # asymmetric interactions
                model_roles[cat_id] = {
                    'category':caps[cat_id],
                    'active_index':learned_indices[tmp],
                    'passive_index':learned_indices[tmp+1]
                }
            tmp+=len(cat_indices)
        with open(pjoin(opt.save_root, 'pit_labels.json'), 'w') as f:
            json.dump(model_roles, f)


    dist.destroy_process_group()


def main(opt):
    world_size = torch.cuda.device_count()
    mp.spawn(
        ddp_train,
        args=(world_size, opt),
        nprocs=world_size,
        join=True
    )

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt_path', type=str, default='checkpoints/ntu_mul/pit/opt.txt', help='Opt path')
    parser.add_argument('--which_epoch', type=str, default="latest", help='Checkpoint that will be used')
    parser.add_argument('--label_path', type=str, default="./data/NTURGBD_multi/test_active_anns.json")
    parser.add_argument('--port', type=str, default='12345')
    parser.add_argument('--label_model', action="store_true", help='')
    parser.add_argument('--save_label', action="store_true", help='')
    args = parser.parse_args()

    parser = TrainCompOptions()
    opt = get_opt(args.opt_path, args.which_epoch, None)
    opt.label_model = args.label_model
    opt.save_label = args.save_label
    opt.label_path = args.label_path
    opt.port = args.port
    opt.is_train = True

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = opt.port
    main(opt)

    if opt.save_label_dir:
        from glob import glob
        merged = {}
        for one in glob(os.path.join(opt.save_label_dir, '*.txt')):
            file_id = one.split('/')[-1].split('.')[0]
            f = open(one)
            merged[file_id] = int(f.read())
            f.close()
        with open('data/NTURGBD_multi/pseudo_labels.json', 'w') as f:
            json.dump(merged, f)



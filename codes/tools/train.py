import os
from os.path import join as pjoin
from os.path import dirname, abspath
import sys
sys.path.append(dirname(dirname(abspath(__file__))))

import utils.paramUtil as paramUtil
from options.train_options import TrainCompOptions
from utils.plot_script import *

from models import MotionInteractionTransformer
from trainers import DDPMMulTrainer
from datasets import Text2MotionMulDataset

from mmcv.runner import get_dist_info, init_dist
from mmcv.parallel import MMDistributedDataParallel
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def build_models(opt, dim_pose):
    encoder = MotionInteractionTransformer(
        input_feats=dim_pose,
        num_frames=opt.max_motion_length,
        num_layers=opt.num_layers,
        latent_dim=opt.latent_dim,
        no_clip=opt.no_clip,
        no_eff=opt.no_eff,
        no_cross_attn=opt.no_cross_attn,
        cap_id=opt.cap_id,
        )
    if opt.pretrained:
        checkpoint = torch.load('checkpoints/t2m/t2m_motiondiffuse/model/latest.tar')
        encoder.load_my_state_dict(checkpoint['encoder'], opt)
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
    dim_word = 300
    mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
    std = np.load(pjoin(opt.data_root, 'Std.npy'))

    train_split_file = pjoin(opt.data_root, 'train_sub.txt')

    encoder = build_models(opt, dim_pose)
    if world_size > 1:
        encoder = MMDistributedDataParallel(
            encoder.cuda(local_rank),
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=False) ## check if unesed parameter exists
    else:
        encoder = encoder.cuda()

    trainer = DDPMMulTrainer(opt, encoder)
    train_dataset = Text2MotionMulDataset(opt, mean, std, train_split_file, opt.times, label_path=opt.label_path)
    trainer.train(train_dataset, local_rank, world_size)

    dist.destroy_process_group()


def main(opt):
    world_size = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = opt.port
    mp.spawn(
        ddp_train,
        args=(world_size, opt),
        nprocs=world_size,
        join=True
    )

if __name__ == '__main__':

    parser = TrainCompOptions()
    opt = parser.parse()

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')

    if opt.dataset_name == 't2m':
        opt.data_root = './data/HumanML3D'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        opt.max_motion_length = 196
    elif opt.dataset_name == 'kit':
        opt.data_root = './data/KIT-ML'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 21
        opt.max_motion_length = 196
    elif opt.dataset_name == 'ntu_mul':
        opt.data_root = './data/NTURGBD_multi'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        opt.max_motion_length = 196
    elif opt.dataset_name == 'multi_pose':
        opt.data_root = './data/MultiPose'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        opt.max_motion_length = 196
    else:
        raise KeyError('Dataset Does Not Exist')
    main(opt)

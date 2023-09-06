import os
from os.path import join as pjoin
from os.path import dirname, abspath
import sys
sys.path.append(dirname(dirname(abspath(__file__))))

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import utils.paramUtil as paramUtil
from options.train_options import TrainCompOptions
from utils.plot_script import *

from models import MotionConsistencyEvalModel
from trainers import DDPMMulTrainer
from datasets import Text2MotionPairDataset

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp


def build_models(opt, dim_pose):
    encoder = MotionConsistencyEvalModel(
        input_feats=dim_pose-4,
        num_frames=opt.max_motion_length,
        num_layers=opt.num_layers,
        latent_dim=opt.latent_dim
        )
    return encoder

def main(opt):

    opt.device = torch.device(0)
    torch.autograd.set_detect_anomaly(True)

    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.meta_dir, exist_ok=True)

    if opt.dataset_name == 't2m':
        radius = 4
        fps = 20
        dim_pose = 263
        kinematic_chain = paramUtil.t2m_kinematic_chain
    elif opt.dataset_name == 'kit':
        radius = 240 * 8
        fps = 12.5
        dim_pose = 251
        kinematic_chain = paramUtil.kit_kinematic_chain
    elif opt.dataset_name == 'ntu_mul':
        radius = 4
        fps = 20
        dim_pose = 263
        kinematic_chain = paramUtil.t2m_kinematic_chain
    elif opt.dataset_name == 'multi_pose':
        radius = 4
        fps = 20
        dim_pose = 263
        kinematic_chain = paramUtil.t2m_kinematic_chain
    else:
        raise KeyError('Dataset Does Not Exist')
    
    opt.meta_dir = pjoin(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name), 'meta')
    mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
    std = np.load(pjoin(opt.meta_dir, 'std.npy'))
    opt.is_train=False


    encoder = build_models(opt, dim_pose)
    encoder = encoder.cuda()
    encoder.load_state_dict(
        torch.load(os.path.join(opt.model_dir, 'best_eval_model.pth'))
    )

    test_split_file = pjoin(opt.data_root, 'test_sub.txt')
    test_dataset = Text2MotionPairDataset(opt, mean.copy(), std.copy(), test_split_file, 1)
    test_loader = DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        num_workers=4,
        pin_memory=False,
        shuffle=False)

    encoder.eval()
    acc_list = []
    acc_per_class = {i:[] for i in range(26)}
    pred_list = []
    gt_list = []
    for i, batch_data in tqdm(enumerate(test_loader)):
        cap_id, motion1, motion2, m_lens, _, label = batch_data
        motion1 = motion1.to(0).float()[:,:,:-4]
        motion2 = motion2.to(0).float()[:,:,:-4]
        with torch.no_grad():
            pred = encoder(motion1, motion2, length=m_lens)
        predicted_classes = pred.max(dim=1).indices.cpu()
        acc_list.extend(predicted_classes.numpy()==label.numpy())
        pred_list.extend(predicted_classes.numpy().tolist())
        gt_list.extend(label.numpy().tolist())
        cap_id = cap_id.numpy()
        for j, one_acc in enumerate(predicted_classes.numpy()==label.numpy()):
            acc_per_class[cap_id[j]].append(one_acc)
        
    print('Accuracy: ', sum(acc_list)/len(acc_list))
    for key in acc_per_class:
        print(key, sum(acc_per_class[key])/len(acc_per_class[key]))


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

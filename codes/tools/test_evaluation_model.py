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

from models import MotionEncoder
from trainers import DDPMMulTrainer
from datasets import Text2MotionMulDataset

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp


def build_models(opt, dim_pose):
    encoder = MotionEncoder(
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
    test_dataset = Text2MotionMulDataset(opt, mean, std, test_split_file, 1, train_eval=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        num_workers=4,
        pin_memory=False,
        shuffle=False)

    encoder.eval()
    acc_list = []
    pred_list = []
    gt_list = []
    for i, batch_data in tqdm(enumerate(test_loader)):
        class_id, motion1, motion2, m_lens, _ = batch_data
        motion1 = motion1.to(0).float()[:,:,:-4]
        motion2 = motion2.to(0).float()[:,:,:-4]
        with torch.no_grad():
            pred, _ = encoder(motion1, motion2, length=m_lens)
        predicted_classes = pred.max(dim=1).indices.cpu()
        acc_list.extend(predicted_classes.numpy()==class_id.numpy())
        pred_list.extend(predicted_classes.numpy().tolist())
        gt_list.extend(class_id.numpy().tolist())
        
    print('Accuracy: ', sum(acc_list)/len(acc_list))
    fig, ax = plt.subplots(figsize=(20, 20))
    cm = confusion_matrix(gt_list, pred_list)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax)
    plt.savefig('result/confusion_matrix.png')


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
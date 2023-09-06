import os
from os.path import join as pjoin
from os.path import dirname, abspath
import sys
sys.path.append(dirname(dirname(abspath(__file__))))

import matplotlib.pyplot as plt
import utils.paramUtil as paramUtil
from options.train_options import TrainCompOptions
from utils.plot_script import *

from models import MotionEncoder, MotionConsistencyEvalModel
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
        input_feats=dim_pose,
        num_frames=opt.max_motion_length,
        num_layers=opt.num_layers,
        latent_dim=opt.latent_dim,
        class_num=26
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
        dim_pose = 4+63+126+66#+4
        kinematic_chain = paramUtil.t2m_kinematic_chain
    elif opt.dataset_name == 'multi_pose':
        radius = 4
        fps = 20
        dim_pose = 263
        kinematic_chain = paramUtil.t2m_kinematic_chain
    else:
        raise KeyError('Dataset Does Not Exist')
        
    mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
    std = np.load(pjoin(opt.data_root, 'Std.npy'))

    train_split_file = pjoin(opt.data_root, 'train_sub.txt')
    val_split_file = pjoin(opt.data_root, 'val_sub.txt')

    encoder = build_models(opt, dim_pose)
    encoder = encoder.cuda()
    opt_encoder = optim.Adam(encoder.parameters(), lr=opt.lr)

    lossFn = nn.CrossEntropyLoss()

    train_dataset = Text2MotionMulDataset(opt, mean.copy(), std.copy(), train_split_file, 1, train_eval=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        num_workers=4,
        pin_memory=False,
        shuffle=True)
    val_dataset = Text2MotionMulDataset(opt, mean.copy(), std.copy(), val_split_file, 1, train_eval=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        num_workers=4,
        pin_memory=False,
        shuffle=False)
    accuracy_train_results = []
    accuracy_val_results = []
    best_acc = 0

    for epoch in range(1, opt.num_epochs):
        encoder.train()
        train_acc_list = []
        for i, batch_data in enumerate(train_loader):
            class_id, motion1, motion2, m_lens, _ = batch_data
            class_id = class_id.to(0)
            motion1 = motion1.to(0).float()[:,:,:dim_pose]
            motion2 = motion2.to(0).float()[:,:,:dim_pose]
            pred, _ = encoder(motion1, motion2, length=m_lens)
            predicted_classes = pred.max(dim=1).indices.cpu()
            train_acc_list.extend(predicted_classes.numpy()==class_id.cpu().numpy())
            loss = lossFn(pred, class_id)
            opt_encoder.zero_grad()
            loss.backward()
            opt_encoder.step()
        encoder.eval()
        val_acc_list = []
        for i, batch_data in enumerate(val_loader):
            class_id, motion1, motion2, m_lens, _ = batch_data
            motion1 = motion1.to(0).float()[:,:,:dim_pose]
            motion2 = motion2.to(0).float()[:,:,:dim_pose]
            with torch.no_grad():
                pred, _ = encoder(motion1, motion2, length=m_lens)
            predicted_classes = pred.max(dim=1).indices.cpu()
            val_acc_list.extend(predicted_classes.numpy()==class_id.numpy())
        accuracy_train_results.append(sum(train_acc_list)/len(train_acc_list))
        accuracy_val_results.append(sum(val_acc_list)/len(val_acc_list))

        if best_acc<accuracy_val_results[-1]:
            best_acc = accuracy_val_results[-1]
            torch.save(encoder.state_dict(), os.path.join(opt.model_dir, 'best_eval_model.pth'))
            print('best acc: ', best_acc)
            print('model saved')

        plt.plot(np.arange(len(accuracy_train_results))+1, accuracy_train_results)
        plt.plot(np.arange(len(accuracy_val_results))+1, accuracy_val_results)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.savefig('result/eval_model_acc.jpg')
        plt.close()
        print(epoch, 'epoch done')
    

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

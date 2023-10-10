"""
Copyright 2023 LINE Corporation

LINE Corporation licenses this file to you under the Apache License,
version 2.0 (the "License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at:

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations
under the License.
"""
import os
from os.path import join as pjoin
from os.path import dirname, abspath
import sys
sys.path.append(dirname(dirname(abspath(__file__))))

import matplotlib.pyplot as plt
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
        
    mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
    std = np.load(pjoin(opt.data_root, 'Std.npy'))

    train_split_file = pjoin(opt.data_root, 'train_sub.txt')
    val_split_file = pjoin(opt.data_root, 'val_sub.txt')

    encoder = build_models(opt, dim_pose)
    encoder = encoder.cuda()
    opt_encoder = optim.Adam(encoder.parameters(), lr=opt.lr/5)

    lossFn = nn.CrossEntropyLoss()

    train_dataset = Text2MotionPairDataset(opt, mean.copy(), std.copy(), train_split_file, 1)
    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        num_workers=4,
        pin_memory=False,
        shuffle=True)
    val_dataset = Text2MotionPairDataset(opt, mean.copy(), std.copy(), val_split_file, 1)
    val_loader = DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        num_workers=4,
        pin_memory=False,
        shuffle=False)
    accuracy_train_results = []
    train_losses = []
    accuracy_val_results = []
    best_acc = 0

    for epoch in range(1, opt.num_epochs):
        encoder.train()
        train_acc_list = []
        train_loss = []
        for i, batch_data in enumerate(train_loader):
            cap_id, motion1, motion2, m_lens, _, label = batch_data
            label = label.to(0)
            motion1 = motion1.to(0).float()[:,:,:-4]
            motion2 = motion2.to(0).float()[:,:,:-4]
            pred = encoder(motion1, motion2, length=m_lens)
            predicted_classes = pred.max(dim=1).indices.cpu()
            train_acc_list.extend(predicted_classes.numpy()==label.cpu().numpy())
            loss = lossFn(pred, label)
            opt_encoder.zero_grad()
            loss.backward()
            opt_encoder.step()
            train_loss.extend([loss.cpu().detach().numpy()]*len(motion1))
        train_losses.append(np.mean(train_loss))
        encoder.eval()
        val_acc_list = []
        for i, batch_data in enumerate(val_loader):
            cap_id, motion1, motion2, m_lens, _, label = batch_data
            motion1 = motion1.to(0).float()[:,:,:-4]
            motion2 = motion2.to(0).float()[:,:,:-4]
            with torch.no_grad():
                pred = encoder(motion1, motion2, length=m_lens)
            predicted_classes = pred.max(dim=1).indices.cpu()
            val_acc_list.extend(predicted_classes.numpy()==label.numpy())
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
        plt.savefig('result/eval_consitency_model_acc.jpg')
        plt.close()

        plt.plot(np.arange(len(train_losses))+1, train_losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig('result/eval_consitency_model_loss.jpg')
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

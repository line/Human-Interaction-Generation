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
import torch
import torch.nn.functional as F
import numpy as np
import os
import json
import random
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from models.transformer import MotionTransformer
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from collections import OrderedDict
from collections import Counter
from utils.utils import print_current_loss
from os.path import join as pjoin
import codecs as cs
import torch.distributed as dist


from mmcv.runner import get_dist_info
from models.gaussian_diffusion import (
    GaussianDiffusion,
    get_named_beta_schedule,
    create_named_schedule_sampler,
    ModelMeanType,
    ModelVarType,
    LossType
)

from datasets import build_dataloader


class DDPMMulTrainer(object):

    def __init__(self, args, encoder):
        self.opt = args
        self.device = args.device
        self.multi = args.multi
        self.with_label = args.label_path is not None
        self.cap_id = args.cap_id
        self.encoder = encoder
        self.diffusion_steps = args.diffusion_steps
        sampler = 'uniform'
        beta_scheduler = 'linear'
        betas = get_named_beta_schedule(beta_scheduler, self.diffusion_steps)
        self.diffusion = GaussianDiffusion(
            betas=betas,
            model_mean_type=ModelMeanType.EPSILON,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE
        )
        self.sampler = create_named_schedule_sampler(sampler, self.diffusion)
        self.sampler_name = sampler

        if args.is_train:
            self.mse_criterion = torch.nn.MSELoss(reduction='none')
        self.to(self.device)

    @staticmethod
    def zero_grad(opt_list):
        for opt in opt_list:
            opt.zero_grad()

    @staticmethod
    def clip_norm(network_list):
        for network in network_list:
            clip_grad_norm_(network.parameters(), 0.5)

    @staticmethod
    def step(opt_list):
        for opt in opt_list:
            opt.step()

    def forward(self, batch_data, eval_mode=False):
        if self.multi:
            caption1, caption2, motion1, motion2, m_lens, _ = batch_data
            motion1 = motion1.detach().to(self.device).float()
            motion2 = motion2.detach().to(self.device).float()
            motion = torch.cat([motion1, motion2], dim=0)
            caption = []
            caption.extend(caption1)
            caption.extend(caption2)

            #self.caption = caption
            #self.motions = motions
            B = motion1.shape[0]
            x_start = motion
            T = x_start.shape[1]
            cur_len = torch.LongTensor([min(T, m_len) for m_len in  m_lens]).to(self.device)
            t, _ = self.sampler.sample(B, x_start.device)
            t = torch.cat([t, t], dim=0)
        
            if not self.with_label:
                """
                (noised_motion1, noised_motion1, noised_motion2, noised_motion2) 
                (caption1, caption2, caption2, caption1)
                (t, t, t, t)
                (cur_len, cur_len, cur_len, cur_len)
                """
                caption.extend(caption2)
                caption.extend(caption1)
                cur_len = torch.cat([cur_len, cur_len, cur_len, cur_len], dim=0)
                forward_twice = True
            else:
                cur_len = torch.cat([cur_len, cur_len], dim=0)
                forward_twice = False
                
            output = self.diffusion.training_losses(
                model=self.encoder,
                x_start=x_start,
                t=t,
                model_kwargs={"text": caption, "length": cur_len},
                forward_twice=forward_twice
            )
            self.real_noise = output['target']
            self.fake_noise = output['pred']
            try:
                src_mask = self.encoder.module.generate_src_mask(T, cur_len)
                self.src_mask = src_mask.to(x_start.device)
            except:
                src_mask = self.encoder.generate_src_mask(T, cur_len)
                self.src_mask = src_mask.to(x_start.device)
        else:
            caption, motions, m_lens = batch_data
            motions = motions.detach().to(self.device).float()

            self.caption = caption
            self.motions = motions
            x_start = motions
            B, T = x_start.shape[:2]
            cur_len = torch.LongTensor([min(T, m_len) for m_len in  m_lens]).to(self.device)
            t, _ = self.sampler.sample(B, x_start.device)
            output = self.diffusion.training_losses(
                model=self.encoder,
                x_start=x_start,
                t=t,
                model_kwargs={"text": caption, "length": cur_len}
            )

            self.real_noise = output['target']
            self.fake_noise = output['pred']
            try:
                self.src_mask = self.encoder.module.generate_src_mask(T, cur_len).to(x_start.device)
            except:
                self.src_mask = self.encoder.generate_src_mask(T, cur_len).to(x_start.device)

    def generate_batch(self, caption1, caption2, m_lens, dim_pose):
        m_lens = torch.cat([m_lens, m_lens], dim=0)
        T = min(m_lens.max(), self.encoder.num_frames)
        if self.cap_id:
            caption = []
            caption.extend(torch.tensor([caption1]))
            caption.extend(torch.tensor([caption2]))
            B = len(caption1)+len(caption2)
            output = self.diffusion.p_sample_loop(
                self.encoder,
                (B, T, dim_pose),
                clip_denoised=False,
                progress=True,
                model_kwargs={
                    'text': caption,
                    'length': m_lens
                })

        else:
            caption = []
            caption.extend(caption1)
            caption.extend(caption2)
            B = len(caption)
            xf_proj, xf_out = self.encoder.encode_text(caption, self.device)
            output = self.diffusion.p_sample_loop(
                self.encoder,
                (B, T, dim_pose),
                clip_denoised=False,
                progress=True,
                model_kwargs={
                    'xf_proj': xf_proj,
                    'xf_out': xf_out,
                    'length': m_lens
                })
        return output

    def generate(self, caption1, caption2, m_lens, dim_pose, batch_size=512):
        N = len(caption1)
        cur_idx = 0
        self.encoder.eval()
        all_output = []
        while cur_idx < N:
            if cur_idx + batch_size >= N:
                batch_caption1 = caption1[cur_idx:]
                batch_caption2 = caption2[cur_idx:]
                batch_m_lens = m_lens[cur_idx:]
            else:
                batch_caption1 = caption1[cur_idx: cur_idx + batch_size]
                batch_caption2 = caption1[cur_idx: cur_idx + batch_size]
                batch_m_lens = m_lens[cur_idx: cur_idx + batch_size]
            
            output = self.generate_batch(batch_caption1, batch_caption2, batch_m_lens, dim_pose)
            B = len(batch_caption1)
            motion1, motion2 = output[:B], output[B:]
            for i in range(B):
                all_output.append([motion1[i], motion2[i]])
            cur_idx += batch_size
        return all_output

    def backward_G(self):
        if self.with_label:
            if self.encoder.module.two_embed:
                loss_init_mot_rec = self.mse_criterion(self.fake_noise[:,0,:4], self.real_noise[:,0,:4]).mean(dim=-1)
                loss_move_mot_rec = self.mse_criterion(self.fake_noise[:,1:], self.real_noise[:,1:]).mean(dim=-1)
                loss_mot_rec = torch.cat([loss_init_mot_rec.unsqueeze(1), loss_move_mot_rec], dim=1)
                loss_mot_rec = (loss_mot_rec * self.src_mask).sum() / self.src_mask.sum()
            else:
                loss_mot_rec = self.mse_criterion(self.fake_noise, self.real_noise).mean(dim=-1)
                loss_mot_rec = (loss_mot_rec * self.src_mask).sum() / self.src_mask.sum()
        else:
            ## loss function for PIT
            loss_init_mot_rec = self.mse_criterion(self.fake_noise[:,0,:4], self.real_noise[:,0,:4]).mean(dim=-1)
            loss_move_mot_rec = self.mse_criterion(self.fake_noise[:,1:], self.real_noise[:,1:]).mean(dim=-1)
            loss_mot_rec = torch.cat([loss_init_mot_rec.unsqueeze(1), loss_move_mot_rec], dim=1)
            B, seq_num = loss_mot_rec.size()
            # (loss_m1_c1 + loss_m2_c1, loss_m1_c2 + loss_m2_c2)
            loss_mot_rec = (loss_mot_rec * self.src_mask).sum(dim=1).view(2, B//2).sum(dim=0)
            # min(loss_m1_c1 + loss_m2_c1, loss_m1_c2 + loss_m2_c2)
            loss_mot_rec = loss_mot_rec.view(2, B//4).min(dim=0).values.sum()/(self.src_mask.sum()/2)

        self.loss_mot_rec = loss_mot_rec
        loss_logs = OrderedDict({})
        loss_logs['loss_mot_rec'] = self.loss_mot_rec.item()
        return loss_logs

    def update(self):
        self.zero_grad([self.opt_encoder])
        loss_logs = self.backward_G()
        self.loss_mot_rec.backward()
        self.clip_norm([self.encoder])
        self.step([self.opt_encoder])

        return loss_logs

    def to(self, device):
        if self.opt.is_train:
            self.mse_criterion.to(device)
        self.encoder = self.encoder.to(device)

    def train_mode(self):
        self.encoder.train()

    def eval_mode(self):
        self.encoder.eval()

    def save(self, file_name, ep, total_it):
        state = {
            'opt_encoder': self.opt_encoder.state_dict(),
            'ep': ep,
            'total_it': total_it
        }
        try:
            state['encoder'] = self.encoder.module.state_dict()
        except:
            state['encoder'] = self.encoder.state_dict()
        torch.save(state, file_name)
        return

    def load(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        if self.opt.is_train:
            self.opt_encoder.load_state_dict(checkpoint['opt_encoder'])
        self.encoder.load_state_dict(checkpoint['encoder'], strict=True)
        return checkpoint['ep'], checkpoint.get('total_it', 0)

    def train(self, train_dataset, rank, world_size):
        self.to(self.device)
        self.opt_encoder = optim.Adam(self.encoder.parameters(), lr=self.opt.lr)
        it = 0
        cur_epoch = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            cur_epoch, it = self.load(model_dir)

        start_time = time.time()

        train_loader = build_dataloader(
            train_dataset,
            rank, world_size,
            samples_per_gpu=self.opt.batch_size,
            drop_last=True,
            workers_per_gpu=4,
            shuffle=True)
            
        logs = OrderedDict()
        mean_losses = []
        for epoch in range(cur_epoch, self.opt.num_epochs):
            self.train_mode()
            for i, batch_data in enumerate(train_loader):
                self.forward(batch_data)
                log_dict = self.update()
                for k, v in log_dict.items():
                    if k not in logs:
                        logs[k] = v
                    else:
                        logs[k] += v
                it += 1
                if it % self.opt.log_every == 0 and rank == 0:
                    mean_loss = OrderedDict({})
                    for tag, value in logs.items():
                        mean_loss[tag] = value / self.opt.log_every
                        mean_losses.append(mean_loss[tag])
                    logs = OrderedDict()
                    print_current_loss(start_time, it, mean_loss, epoch, inner_iter=i)

                    plt.plot(np.arange(len(mean_losses)), mean_losses)
                    plt.xlabel("iter")
                    plt.ylabel("loss")
                    plt.savefig('result/result_loss.jpg')
                if it % self.opt.save_latest == 0 and rank == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            if rank == 0:
                self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            if epoch % self.opt.save_every_e == 0 and rank == 0:
                self.save(pjoin(self.opt.model_dir, 'ckpt_e%03d.tar'%(epoch)),
                            epoch, total_it=it)

    def label_batch(self, batch_data,learned_indices,t,label_mode=False):
        
        caption1, caption2, motion1, motion2, m_lens, file_id = batch_data
        if learned_indices is not None:
            cap1 = [learned_indices[cap] for cap in caption1[0].numpy()]
            cap2 = [learned_indices[cap] for cap in caption2[0].numpy()]
            active_indices = np.argmin([cap1, cap2], axis=0)

        motion1 = motion1.detach().to(self.device).float()
        motion2 = motion2.detach().to(self.device).float()
        motion = torch.cat([motion1, motion2], dim=0)
        caption = []
        caption.extend(caption1)
        caption.extend(caption2)
        caption.extend(caption2)
        caption.extend(caption1)

        B = motion1.shape[0]
        x_start = motion
        T = x_start.shape[1]
        cur_len = torch.LongTensor([min(T, m_len) for m_len in  m_lens]).to(self.device)
        t = torch.from_numpy(np.ones(B)*t).long().to(x_start.device)
        t = torch.cat([t, t], dim=0)
        cur_len = torch.cat([cur_len, cur_len, cur_len, cur_len], dim=0)
        """
        (noised_motion1, noised_motion1, noised_motion2, noised_motion2) 
        (caption1, caption2, caption2, caption1)
        (t, t, t, t)
        (cur_len, cur_len, cur_len, cur_len)
        """
        output = self.diffusion.training_losses(
            model=self.encoder,
            x_start=x_start,
            t=t,
            model_kwargs={"text": caption, "length": cur_len},
            forward_twice=True
        )
        self.real_noise = output['target']
        self.fake_noise = output['pred']
        try:
            src_mask = self.encoder.module.generate_src_mask(T, cur_len)
            self.src_mask = src_mask.to(x_start.device)
        except:
            src_mask = self.encoder.generate_src_mask(T, cur_len)
            self.src_mask = src_mask.to(x_start.device)

        loss_init_mot_rec = self.mse_criterion(self.fake_noise[:,0,:4], self.real_noise[:,0,:4]).mean(dim=-1)
        loss_move_mot_rec = self.mse_criterion(self.fake_noise[:,1:], self.real_noise[:,1:]).mean(dim=-1)
        loss_mot_rec = torch.cat([loss_init_mot_rec.unsqueeze(1), loss_move_mot_rec], dim=1)
        B, seq_num = loss_mot_rec.size()
        loss_mot_rec = (loss_mot_rec * self.src_mask).sum(dim=1).view(2, B//2).sum(dim=0)
        
        loss_diff = (loss_mot_rec.view(2, B//4).max(dim=0).values-loss_mot_rec.view(2, B//4).min(dim=0).values).sum()
        result = loss_mot_rec.view(2, B//4).min(dim=0).indices.cpu().numpy()
        
        if learned_indices is not None:
            outs = []
            for i, res in enumerate(result):
                if (res==0 and active_indices[i]==0) or (res==1 and active_indices[i]==1):
                    outs.append(0)
                else:
                    outs.append(1)
            if label_mode:
                return outs, file_id
            else:
                return np.array(outs)
        else:
            active_list = {}
            for i, res in enumerate(result):
                cap1, cap2 = int(caption1[0].numpy()[i]), int(caption2[0].numpy()[i])
                pair_key = str(cap1)+'_'+str(cap2)
                if pair_key not in active_list:
                    active_list[pair_key] = []
                if res==0:
                    active_list[pair_key].append(str(cap1)+'_'+str(cap2))
                else:
                    active_list[pair_key].append(str(cap2)+'_'+str(cap1))
            return active_list


    def eval_data(self, train_dataset, rank, world_size, learned_indices, max_class_num=42, save_dir=None):
        self.to(self.device)
        start_time = time.time()

        train_loader = build_dataloader(
            train_dataset,
            rank, world_size,
            samples_per_gpu=self.opt.batch_size,
            drop_last=False,
            workers_per_gpu=4,
            shuffle=False)
            
        logs = OrderedDict()
        self.eval_mode()

        if learned_indices is None:
            with torch.no_grad():
                merged_active_list = {}
                for t in tqdm(range(830,940,30)):
                    for i, batch_data in enumerate(train_loader):
                        for _ in range(5):
                            active_list = self.label_batch(batch_data,None,t=t)
                            for key in active_list:
                                if key not in merged_active_list:
                                    merged_active_list[key] = []
                                merged_active_list[key].extend(active_list[key])
                learned_indices = []
                for i in range(max_class_num+1):
                    if str(i-1)+'_'+str(i) in merged_active_list:
                        continue
                    elif str(i)+'_'+str(i+1) in merged_active_list:
                        num1, num2 = Counter(merged_active_list[str(i)+'_'+str(i+1)]).most_common()[0][0].split('_')
                        learned_indices.append(int(num1))
                        learned_indices.append(int(num2))
                    else:
                        learned_indices.append(i)
            return learned_indices
        else:
            with torch.no_grad():
                for i, batch_data in tqdm(enumerate(train_loader)):
                    file_id2label = {}
                    for t in range(830,940,30):
                        for _ in range(41):
                            outs, file_ids = self.label_batch(batch_data,learned_indices,t=t,label_mode=True)
                            for iii in range(len(file_ids)):
                                if file_ids[iii] not in file_id2label:
                                    file_id2label[file_ids[iii]] = []
                                file_id2label[file_ids[iii]].append(outs[iii])
                    for key in file_id2label:
                        label = Counter(file_id2label[key]).most_common()[0][0]
                        f = open(os.path.join(save_dir, key+'.txt'), 'w')
                        f.write(str(label))
                        f.close()

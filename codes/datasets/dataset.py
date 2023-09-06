import torch
from torch.utils import data
import numpy as np
import os
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm


class Text2MotionDataset(data.Dataset):
    """Dataset for Text2Motion generation task.

    """
    def __init__(self, opt, mean, std, split_file, times=1, w_vectorizer=None, eval_mode=False):
        self.opt = opt
        self.max_length = 20
        self.times = times
        self.w_vectorizer = w_vectorizer
        self.eval_mode = eval_mode
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24
        if self.opt.dataset_name =='t2m':
            min_motion_len = 40
        elif self.opt.dataset_name =='kit':
            min_motion_len = 24
        elif self.opt.dataset_name =='ntu_mul':
            min_motion_len = 20
        joints_num = opt.joints_num

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if (len(motion[1])) < min_motion_len or (len(motion[1]) >= 200):
                    print('!')
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                            if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                continue
                            new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                            while new_name in data_dict:
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                            data_dict[new_name] = {'motion': n_motion,
                                                    'length': len(n_motion),
                                                    'text':[text_dict]}
                            new_name_list.append(new_name)
                            length_list.append(len(n_motion))

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion[1]),
                                       'text':text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion[1]))
            except:
                # Some motion may not exist in KIT dataset
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        if opt.is_train:
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4: 4 + (joints_num - 1) * 3] = std[4: 4 + (joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = std[4 + (joints_num - 1) * 3: 4 + (
                        joints_num - 1) * 9] / 1.0
            # local_velocity (B, seq_len, joint_num*3)
            std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = std[
                                                                                       4 + (joints_num - 1) * 9: 4 + (
                                                                                                   joints_num - 1) * 9 + joints_num * 3] / 1.0
            # foot contact (B, seq_len, 4)
            if self.opt.dataset_name !='ntu_mul':
                std[4 + (joints_num - 1) * 9 + joints_num * 3:] = std[
                                                              4 + (joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias
            else:
                std[4 + (joints_num - 1) * 9 + joints_num * 3: 4 + (joints_num - 1) * 9 + joints_num * 3 + 4] = std[4 + (joints_num - 1) * 9 + joints_num * 3: 4 + (joints_num - 1) * 9 + joints_num * 3 + 4].mean()  / opt.feat_bias
                std[4 + (joints_num - 1) * 9 + joints_num * 3 + 4:] = std[4 + (joints_num - 1) * 9 + joints_num * 3 + 4:].mean() / 1.0

            np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
            np.save(pjoin(opt.meta_dir, 'std.npy'), std)

        self.mean = mean[:-3]
        self.std = std[:-3]
        self.init_mean = mean[-3:]
        self.init_std = std[-3:]
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def real_len(self):
        return len(self.data_dict)

    def __len__(self):
        return self.real_len() * self.times

    def __getitem__(self, item):
        idx = item % self.real_len()
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        if True:
            num_frames = 60
            nframes = motion.shape[1]-1 ## 最後はinit_posを入れているので
            if num_frames > nframes:
                # adding the last frame until done
                ntoadd = max(0, num_frames - nframes)
                lastframe = nframes - 1
                padding = lastframe * np.ones(ntoadd, dtype=int)
                frame_ix = np.concatenate((np.arange(0, nframes),
                                            padding, [nframes]))
            else:
                step_max = (nframes - 1) // (num_frames - 1)
                step = 1
                lastone = step * (num_frames - 1)
                shift_max = nframes - lastone - 1
                shift = random.randint(0, max(0, shift_max - 1))
                frame_ix = shift + np.arange(0, lastone + 1, step)
                frame_ix = np.concatenate((frame_ix, [nframes]))
            motion1, motion2 = motion[0][frame_ix], motion[1][frame_ix]
        else:
            if m_length >= self.opt.max_motion_length:
                idx = random.randint(0, len(motion) - max_motion_length)
                motion = motion[idx: idx + max_motion_length]
            else:
                padding_len = max_motion_length - m_length
                D = motion.shape[1]
                padding_zeros = np.zeros((padding_len, D))
                motion = np.concatenate((motion, padding_zeros), axis=0)
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption = text_data['caption']


        #assert len(motion) == max_motion_length
        "Z Normalization"
        motion1[:-1] = (motion1[:-1] - self.mean) / self.std
        motion2[:-1] = (motion2[:-1] - self.mean) / self.std
        motion1[-1,:3] = (motion1[-1,:3] - self.init_mean) / self.init_std
        motion2[-1,:3] = (motion2[-1,:3] - self.init_mean) / self.init_std

        if self.eval_mode:
            tokens = text_data['tokens']
            if len(tokens) < self.opt.max_text_len:
                # pad with "unk"
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
                tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
            else:
                # crop
                tokens = tokens[:self.opt.max_text_len]
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
            pos_one_hots = []
            word_embeddings = []
            for token in tokens:
                word_emb, pos_oh = self.w_vectorizer[token]
                pos_one_hots.append(pos_oh[None, :])
                word_embeddings.append(word_emb[None, :])
            pos_one_hots = np.concatenate(pos_one_hots, axis=0)
            word_embeddings = np.concatenate(word_embeddings, axis=0)
            return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length
        return caption, motion1, motion2, m_length

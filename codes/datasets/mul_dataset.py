import torch
from torch.utils import data
import numpy as np
import os
from os.path import join as pjoin
import json
import random
import codecs as cs
from tqdm import tqdm
from data.NTURGBD_multi.language_labels import ntu_action_multi_enumerator

caps = []
cap2classid = {}
for class_id, key in enumerate(ntu_action_multi_enumerator):
    caps.extend(ntu_action_multi_enumerator[key])
    cap2classid[ntu_action_multi_enumerator[key][0]] = class_id
cap2key = {caps[i]:i for i in range(len(caps))}


class Text2MotionMulDataset(data.Dataset):
    """Dataset for Text2Motion generation task.

    """
    def __init__(self, opt, mean, std, split_file, times=1, w_vectorizer=None, eval_mode=False, label_path=None, train_eval=False):
        self.opt = opt
        self.max_length = 20
        self.times = times
        self.cap_id = opt.cap_id
        self.cap_same = opt.cap_same
        self.train_eval = train_eval
        self.w_vectorizer = w_vectorizer

        self.eval_mode = eval_mode
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24
        if self.opt.dataset_name =='t2m':
            min_motion_len = 40
        elif self.opt.dataset_name =='kit':
            min_motion_len = 24
        elif self.opt.dataset_name =='ntu_mul':
            min_motion_len = 20
        elif self.opt.dataset_name =='multi_pose':
            min_motion_len = 20
        joints_num = opt.joints_num

        self.with_label = label_path is not None
        if self.with_label:
            with open(label_path) as f:
                self.label = json.load(f)

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
                if len(motion.shape)==2:
                    current_motion_len = len(motion)
                else:
                    current_motion_len = len(motion[1])

                if current_motion_len < min_motion_len or (current_motion_len >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        captions = line_split[0].split('_')
                        if len(captions)==1:
                            captions.extend(captions)
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['captions'] = captions
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            ## human ml3d only
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
                                       'length': current_motion_len,
                                       'text':text_data}
                    new_name_list.append(name)
                    length_list.append(current_motion_len)
            except:
                # Some motion may not exist in KIT dataset
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        if self.opt.limit_data_num!=-1:
            np.random.seed(0)
            all_indices = np.arange(len(name_list))
            np.random.shuffle(all_indices)
            name_list = [name_list[ind] for ind in all_indices[:self.opt.limit_data_num]]
            length_list = [length_list[ind] for ind in all_indices[:self.opt.limit_data_num]]
            data_dict = {key:data_dict[key] for key in name_list}

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
            np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
            np.save(pjoin(opt.meta_dir, 'std.npy'), std)
            print('mean and std, saved.')
        
        self.mean = mean[:-4]
        self.std = std[:-4]
        self.init_mean = mean[-4:]
        self.init_std = std[-4:]
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
        file_id = self.name_list[idx]
        data = self.data_dict[file_id]
        motion, m_length, text_list = data['motion'], data['length'], data['text']

        num_frames = 90
        nframes = motion.shape[1]-1 
        if num_frames > nframes:
            ntoadd = max(0, num_frames - nframes)
            lastframe = nframes - 1
            padding = lastframe * np.ones(ntoadd, dtype=int)
            frame_ix = np.concatenate(([nframes], np.arange(0, nframes),
                                        padding))
        else:
            step_max = (nframes - 1) // (num_frames - 1)
            step = 1
            lastone = step * (num_frames - 1)
            shift_max = nframes - lastone - 1
            shift = random.randint(0, max(0, shift_max - 1))
            frame_ix = shift + np.arange(0, lastone + 1, step)
            frame_ix = np.concatenate(([nframes], frame_ix))
        motion1, motion2 = motion[0][frame_ix], motion[1][frame_ix]

        "Z Normalization"
        motion1[1:] = (motion1[1:] - self.mean) / self.std
        motion2[1:] = (motion2[1:] - self.mean) / self.std
        motion1[0,:4] = (motion1[0,:4] - self.init_mean) / self.init_std
        motion2[0,:4] = (motion2[0,:4] - self.init_mean) / self.init_std

        # Randomly select a caption
        text_data = random.choice(text_list)
        caption1, caption2 = text_data['captions']
        if self.cap_id:
            caption1 = [cap2key[caption1]]
            caption2 = [cap2key[caption2]]
        elif self.cap_same:
            caption2 = caption1

        if self.eval_mode:
            if True:
                return cap2classid[caption1], motion1, motion2, m_length, file_id
            else:
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
        
        if self.train_eval:
            return cap2classid[caption1], motion1, motion2, m_length, file_id
        elif self.with_label:
            label = self.label[file_id]
            if label==0:
                return caption1, caption2, motion1, motion2, m_length, file_id
            else:
                return caption1, caption2, motion2, motion1, m_length, file_id
        else:
            return caption1, caption2, motion1, motion2, m_length, file_id


def trim_motion(m, length):
    start = random.randint(0, len(m)-length)
    return m[start:start+length][None,:]

class Text2MotionPairDataset(data.Dataset):
    """Dataset for training model for evaluating mutual consistency

    """
    def __init__(self, opt, mean, std, split_file, times=1, w_vectorizer=None, eval_mode=False, label_path=None):
        self.opt = opt
        self.max_length = 20
        self.times = times
        self.cap_id = opt.cap_id
        self.cap_same = opt.cap_same
        self.w_vectorizer = w_vectorizer

        self.eval_mode = eval_mode
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24
        if self.opt.dataset_name =='t2m':
            min_motion_len = 40
        elif self.opt.dataset_name =='kit':
            min_motion_len = 24
        elif self.opt.dataset_name =='ntu_mul':
            min_motion_len = 20
        elif self.opt.dataset_name =='multi_pose':
            min_motion_len = 20
        joints_num = opt.joints_num

        self.with_label = label_path is not None
        if self.with_label:
            with open(label_path) as f:
                self.label = json.load(f)

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        self.cate2names = {}
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if len(motion.shape)==2:
                    current_motion_len = len(motion)
                else:
                    current_motion_len = len(motion[1])

                if current_motion_len < min_motion_len or (current_motion_len >= 200):
                    print('!')
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        captions = line_split[0].split('_')
                        if len(captions)==1:
                            captions.extend(captions)
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['captions'] = captions
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            ## human ml3d only
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
                                       'length': current_motion_len,
                                       'text':text_data}
                    if captions[0] not in self.cate2names:
                        self.cate2names[captions[0]] = []
                    self.cate2names[captions[0]].append(name)
                    new_name_list.append(name)
                    length_list.append(current_motion_len)
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
            np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
            np.save(pjoin(opt.meta_dir, 'std.npy'), std)
            print('mean and std, saved.')
        
        self.mean = mean[:-4]
        self.std = std[:-4]
        self.init_mean = mean[-4:]
        self.init_std = std[-4:]
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
        if random.random()>0.5:
            dummy_label = 0
        else:
            dummy_label = 1

        idx = item % self.real_len()
        file_id = self.name_list[idx]
        data = self.data_dict[file_id]
        motion, m_length, text_list = data['motion'], data['length'], data['text']

        if dummy_label==1:
            while True:
                motion = random.choice(motion)
                dummy_file_id = random.choice(self.cate2names[text_list[0]['captions'][0]])
                dummy_data = self.data_dict[dummy_file_id]
                dummy_motion, dummy_length = random.choice(data['motion']), data['length']
                m_length = min(m_length, dummy_length)
                motion = np.concatenate([trim_motion(motion, m_length), trim_motion(dummy_motion, m_length)], axis=0)
                if dummy_file_id!=file_id:
                    break

        num_frames = 90
        nframes = motion.shape[1]-1 
        if num_frames > nframes:
            # adding the last frame until done
            ntoadd = max(0, num_frames - nframes)
            lastframe = nframes - 1
            padding = lastframe * np.ones(ntoadd, dtype=int)
            frame_ix = np.concatenate(([nframes], np.arange(0, nframes),
                                        padding))
        else:
            step_max = (nframes - 1) // (num_frames - 1)
            step = 1
            lastone = step * (num_frames - 1)
            shift_max = nframes - lastone - 1
            shift = random.randint(0, max(0, shift_max - 1))
            frame_ix = shift + np.arange(0, lastone + 1, step)
            frame_ix = np.concatenate(([nframes], frame_ix))
        motion1, motion2 = motion[0][frame_ix], motion[1][frame_ix]

        "Z Normalization"
        motion1[1:] = (motion1[1:] - self.mean) / self.std
        motion2[1:] = (motion2[1:] - self.mean) / self.std
        motion1[0,:4] = (motion1[0,:4] - self.init_mean) / self.init_std
        motion2[0,:4] = (motion2[0,:4] - self.init_mean) / self.init_std

        # Randomly select a caption
        text_data = random.choice(text_list)
        caption1, caption2 = text_data['captions']
        if self.cap_id:
            caption1 = [cap2key[caption1]]
            caption2 = [cap2key[caption2]]
        elif self.cap_same:
            caption2 = caption1

        
        return cap2classid[caption1], motion1, motion2, m_length, file_id, dummy_label

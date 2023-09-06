import torch
import pickle
from utils.get_opt import get_opt
from models import MotionTransformer, MotionEncoder, MotionConsistencyEvalModel
from torch.utils.data import Dataset, DataLoader
from os.path import join as pjoin
from tqdm import tqdm
import numpy as np
from .evaluator_models import *
import os
import codecs as cs
import random
from torch.utils.data._utils.collate import default_collate
from data.NTURGBD_multi.language_labels import ntu_action_multi_enumerator

caps = []
cap2classid = {}
for class_id, key in enumerate(ntu_action_multi_enumerator):
    caps.extend(ntu_action_multi_enumerator[key])
    cap2classid[ntu_action_multi_enumerator[key][0]] = class_id

cap2key = {caps[i]:i for i in range(len(caps))}

class EvaluationDataset(Dataset):

    def __init__(self, opt, trainer, dataset, w_vectorizer, mm_num_repeats, generated=None, mm_generated=None):
        
        self.max_motion_length = 90
        if generated is None or mm_generated is None:
            dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True)
            epoch, it = trainer.load(pjoin(opt.model_dir, opt.which_epoch + '.tar'))

            generated_motion = []

            trainer.eval_mode()
            trainer.to(opt.device)

            # Pre-process all target captions
            mm_generated_motions_per_category = {i:0 for i in range(len(ntu_action_multi_enumerator))}
            mm_indices = []
            all_caption1 = []
            all_caption2 = []
            all_caption_id = []
            all_m_lens = []
            all_data = []
            with torch.no_grad():
                for i, data in tqdm(enumerate(dataloader)):
                    cap_id, caption1, caption2, motion1, motion2, m_lens = data
                    all_data.append(data)
                    if mm_generated_motions_per_category[cap_id.numpy()[0]]<=mm_num_repeats:
                        is_mm = True
                        mm_indices.append(i)
                        mm_generated_motions_per_category[cap_id.numpy()[0]]+=1
                    else:
                        is_mm = False

                    if isinstance(m_lens, int):
                        m_lens = torch.LongTensor([m_lens]).to(opt.device)
                    else:
                        m_lens = m_lens.to(opt.device)
                    all_m_lens.append(m_lens)
                    if opt.cap_id:
                        all_caption1.extend(caption1[0])
                        all_caption2.extend(caption2[0])
                    else:
                        all_caption1.extend(caption1)
                        all_caption2.extend(caption2)
                    all_caption_id.extend(cap_id)

            all_m_lens = torch.stack(all_m_lens)
            # Generate all sequences
            with torch.no_grad():
                all_pred_motions = trainer.generate(all_caption1, all_caption2, all_m_lens, opt.dim_pose)
            
            mm_generated_motions_per_category = {i:[] for i in range(len(ntu_action_multi_enumerator))}
            gt_mm_generated_motions_per_category = {i:[] for i in range(len(ntu_action_multi_enumerator))}
            with torch.no_grad():
                for i, data_dummy in tqdm(enumerate(dataloader)):
                    data = all_data[i]
                    cap_id, caption1, caption2, motion1, motion2, m_lens = data

                    if i in mm_indices:
                        is_mm = True
                    else:
                        is_mm = False
                    #m_lens = max(m_lens // opt.unit_length * opt.unit_length, min_mov_length * opt.unit_length)
                    #m_lens = min(m_lens, self.max_motion_length)
                    if isinstance(m_lens, int):
                        m_lens = torch.LongTensor([m_lens]).to(opt.device)
                    else:
                        m_lens = m_lens.to(opt.device)
                    
                    m_len = m_lens[0].item()
                    pred_motions1, pred_motions2 = all_pred_motions[i]
                    pred_motions1, pred_motions2 = pred_motions1[:m_lens[0].item()], pred_motions2[:m_lens[0].item()]
                        
                    assert pred_motions1.shape[0] == m_lens[0].item()
                    sub_dict = {'motion1': pred_motions1.cpu().numpy(),
                                'motion2': pred_motions2.cpu().numpy(),
                                'length': pred_motions1.shape[0],
                                'caption1': caption1[0],
                                'caption2': caption2[0],
                                'cap_id': cap_id[0]}
                    generated_motion.append(sub_dict)
                    
                    if is_mm:
                        mm_generated_motions_per_category[cap_id.numpy()[0]].append({
                            'motion1': pred_motions1.cpu().numpy(),
                            'motion2': pred_motions2.cpu().numpy(),
                            'length': m_lens[0].item()
                        })
                        gt_mm_generated_motions_per_category[cap_id.numpy()[0]].append({
                            'motion1': motion1[0,:m_lens[0].item()].cpu().numpy(),
                            'motion2': motion2[0,:m_lens[0].item()].cpu().numpy(),
                            'length': motion1[0,:m_lens[0].item()].shape[0]
                        })
            mm_generated_motions = [{
                'cap_id':cap_id,
                'mm_motions':motions
            } for cap_id, motions in mm_generated_motions_per_category.items()]
            gt_mm_generated_motions = [{
                'cap_id':cap_id,
                'mm_motions':motions
            } for cap_id, motions in gt_mm_generated_motions_per_category.items()]
            self.generated_motion = generated_motion
            self.mm_generated_motion = mm_generated_motions
            self.gt_mm_generated_motion = gt_mm_generated_motions
        else:
            with open(generated, 'rb') as f:
                self.generated_motion = pickle.load(f)
            with open(mm_generated, 'rb') as f:
                data = pickle.load(f)
                self.mm_generated_motion = [{'cap_id':i,'mm_motions':[]} for i in range(26)]
                for one in data:
                    self.mm_generated_motion[one['cap_id']]['mm_motions'].append(one['mm_motions'])
                print([self.mm_generated_motion[i]['mm_motions'] for i in range(len(self.mm_generated_motion))])


    def __len__(self):
        return len(self.generated_motion)


    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion1, motion2, m_length, caption1, caption2, cap_id = data['motion1'], data['motion2'], data['length'], data['caption1'], data['caption2'], data['cap_id']
        motion1 = np.concatenate([motion1[1:], motion1[0][None, :]], axis=0)
        motion2 = np.concatenate([motion2[1:], motion2[0][None, :]], axis=0)
        # currently 60 frame only
        nframes = m_length-1
        if nframes < self.max_motion_length:
            # adding the last frame until done
            ntoadd = max(0, self.max_motion_length - nframes)
            lastframe = nframes - 1
            padding = lastframe * np.ones(ntoadd, dtype=int)
            frame_ix = np.concatenate(([nframes], np.arange(0, nframes),
                                            padding))
        else:
            step_max = (nframes - 1) // (self.max_motion_length - 1)
            step = 1
            lastone = step * (self.max_motion_length - 1)
            shift_max = nframes - lastone - 1
            shift = random.randint(0, max(0, shift_max - 1))
            frame_ix = shift + np.arange(0, lastone + 1, step)
            frame_ix = np.concatenate(([nframes], frame_ix))
        motion1, motion2 = motion1[frame_ix], motion2[frame_ix]
        return cap_id, caption1, caption2, motion1, motion2, m_length


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


'''For use of training text motion matching model, and evaluations'''
class Text2MotionDatasetV2(Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.cap_same = opt.cap_same
        self.cap_id = opt.cap_id
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        #min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24
        min_motion_len = 20

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():#[:601]:
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
                    new_name_list.append(name)
                    length_list.append(current_motion_len)
            except:
                # Some motion may not exist in KIT dataset
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean[:-4]
        self.std = std[:-4]
        self.init_mean = mean[-4:]
        self.init_std = std[-4:]
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption1, caption2 = text_data['captions']
        target_class_id = cap2classid[caption1]
        if self.cap_id:
            caption1 = [cap2key[caption1]]
            caption2 = [cap2key[caption2]]
        elif self.cap_same:
            caption2 = caption1
        #caption, tokens = text_data['caption'], text_data['tokens']

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
        return target_class_id, caption1, caption2, motion1, motion2, m_length


def get_dataset_motion_loader(opt_path, batch_size, device, split_file):
    opt = get_opt(opt_path, 'latest', device)

    # Configurations of T2M dataset and KIT dataset is almost the same
    if opt.dataset_name == 't2m' or opt.dataset_name == 'kit' or opt.dataset_name == 'ntu_mul':
        print('Loading dataset %s ...' % opt.dataset_name)

        mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
        std = np.load(pjoin(opt.meta_dir, 'std.npy'))

        w_vectorizer = None#WordVectorizer('./data/glove', 'our_vab')
        split_file = pjoin(opt.data_root, split_file)
        dataset = Text2MotionDatasetV2(opt, mean.copy(), std.copy(), split_file, w_vectorizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    else:
        raise KeyError('Dataset not Recognized !!')

    print('Ground Truth Dataset Loading Completed!!!')
    return dataloader, dataset


class MMGeneratedDataset(Dataset):
    def __init__(self, opt, mm_generated_motion, w_vectorizer):
        self.opt = opt
        self.dataset = mm_generated_motion
        self.w_vectorizer = w_vectorizer
        self.max_motion_length = 90

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        mm_motions = data['mm_motions']
        m_lens = []
        motions1 = []
        motions2 = []
        for mm_motion in mm_motions:
            m_lens.append(mm_motion['length'])
            motion1, motion2 = mm_motion['motion1'], mm_motion['motion2']
            motion1 = np.concatenate([motion1[1:], motion1[0][None, :]], axis=0)
            motion2 = np.concatenate([motion2[1:], motion2[0][None, :]], axis=0)

            nframes = mm_motion['length']-1
            if nframes < self.max_motion_length:
                # adding the last frame until done
                ntoadd = max(0, self.max_motion_length - nframes)
                lastframe = nframes - 1
                padding = lastframe * np.ones(ntoadd, dtype=int)
                frame_ix = np.concatenate(([nframes], np.arange(0, nframes),
                                                padding))
            else:
                step_max = (nframes - 1) // (self.max_motion_length - 1)
                step = 1
                lastone = step * (self.max_motion_length - 1)
                shift_max = nframes - lastone - 1
                shift = random.randint(0, max(0, shift_max - 1))
                frame_ix = shift + np.arange(0, lastone + 1, step)
                frame_ix = np.concatenate(([nframes], frame_ix))
            motion1, motion2 = motion1[frame_ix], motion2[frame_ix]
            motions1.append(motion1[None, :])
            motions2.append(motion2[None, :])
        m_lens = np.array(m_lens, dtype=np.int)
        motions1 = np.concatenate(motions1, axis=0)
        motions2 = np.concatenate(motions2, axis=0)
        sort_indx = np.argsort(m_lens)[::-1].copy()
        # print(m_lens)
        # print(sort_indx)
        # print(m_lens[sort_indx])
        m_lens = m_lens[sort_indx]
        motions1 = motions1[sort_indx]
        motions2 = motions2[sort_indx]
        return motions1, motions2, m_lens



def get_motion_loader(opt, batch_size, trainer, ground_truth_dataset, mm_num_repeats):

    if opt.dataset_name == 't2m' or opt.dataset_name == 'kit' or opt.dataset_name == 'ntu_mul':
        w_vectorizer = None
    else:
        raise KeyError('Dataset not recognized!!')
    print('Generating %s ...' % opt.name)

    dataset = EvaluationDataset(opt, trainer, ground_truth_dataset, w_vectorizer, mm_num_repeats)
    mm_dataset = MMGeneratedDataset(opt, dataset.mm_generated_motion, w_vectorizer)
    gt_mm_dataset = MMGeneratedDataset(opt, dataset.gt_mm_generated_motion, w_vectorizer)

    motion_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, num_workers=4)
    mm_motion_loader = DataLoader(mm_dataset, batch_size=1, num_workers=1)
    gt_mm_motion_loader = DataLoader(gt_mm_dataset, batch_size=1, num_workers=1)

    print('Generated Dataset Loading Completed!!!')

    return motion_loader, mm_motion_loader, gt_mm_motion_loader


def build_models(opt):
    motion_enc = MotionEncoder(
        input_feats=opt.dim_pose-4,
        num_frames=opt.max_motion_length,
        num_layers=opt.num_layers,
        latent_dim=opt.latent_dim
    )
    motion_enc.load_state_dict(torch.load('checkpoints/ntu_mul/eval_model/model/best_eval_model.pth'))

    consistency_eval_model = MotionConsistencyEvalModel(
        input_feats=opt.dim_pose-4,
        num_frames=opt.max_motion_length,
        num_layers=opt.num_layers,
        latent_dim=opt.latent_dim
        )
    consistency_eval_model.load_state_dict(torch.load('checkpoints/ntu_mul/consistency_eval_model/model/best_eval_model.pth'))
    return motion_enc, consistency_eval_model


class EvaluatorModelWrapper(object):

    def __init__(self, opt):

        if opt.dataset_name == 't2m' or opt.dataset_name == 'ntu_mul':
            opt.dim_pose = 263
        elif opt.dataset_name == 'kit':
            opt.dim_pose = 251
        else:
            raise KeyError('Dataset not Recognized!!!')

        opt.dim_word = 300
        opt.max_motion_length = 196
        opt.dim_motion_hidden = 1024
        opt.max_text_len = 20
        opt.dim_text_hidden = 512
        opt.dim_coemb_hidden = 512

        #self.text_encoder, self.motion_encoder, self.movement_encoder = build_models(opt)
        self.encoder, self.consistency_eval_model = build_models(opt)
        self.opt = opt
        self.device = opt.device

        self.encoder.to(opt.device)
        self.consistency_eval_model.to(opt.device)

        self.encoder.eval()
        self.consistency_eval_model.eval()

    # Please note that the results does not following the order of inputs
    def get_co_embeddings(self, word_embs, pos_ohot, cap_lens, motions, m_lens):
        with torch.no_grad():
            word_embs = word_embs.detach().to(self.device).float()
            pos_ohot = pos_ohot.detach().to(self.device).float()
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            '''Movement Encoding'''
            movements = self.movement_encoder(motions[..., :-4]).detach()
            m_lens = m_lens // self.opt.unit_length
            motion_embedding = self.motion_encoder(movements, m_lens)

            '''Text Encoding'''
            text_embedding = self.text_encoder(word_embs, pos_ohot, cap_lens)
            text_embedding = text_embedding[align_idx]
        return text_embedding, motion_embedding

    # Please note that the results does not following the order of inputs
    def get_motion_embeddings(self, motions1, motions2, m_lens, keep_dim=False):
        with torch.no_grad():
            if keep_dim:
                motions1 = motions1.detach().to(self.device).float()
                motions2 = motions2.detach().to(self.device).float()
            else:
                motions1 = motions1.detach().to(self.device).float()[:,:,:-4]
                motions2 = motions2.detach().to(self.device).float()[:,:,:-4]
            fin_motion_embedding, motion_embedding = self.encoder(motions1, motions2, length=m_lens)
            consitency_embedding = self.consistency_eval_model(motions1, motions2, length=m_lens)

        return fin_motion_embedding, motion_embedding, consitency_embedding

import os
import shutil
import torch
import argparse
from os.path import join as pjoin
from os.path import dirname, abspath
import sys
sys.path.append(dirname(dirname(abspath(__file__))))

import utils.paramUtil as paramUtil
from torch.utils.data import DataLoader
from utils.plot_script import *
from utils.get_opt import get_opt
from datasets.evaluator_models import MotionLenEstimatorBiGRU

from trainers import DDPMTrainer, DDPMMulTrainer
from models import MotionInteractionTransformer, MotionTransformer
from utils.word_vectorizer import WordVectorizer, POS_enumerator
from utils.utils import *
from utils.motion_process import recover_from_ric, recover_from_ric2
from data.NTURGBD_multi.language_labels import ntu_action_multi_enumerator

caps = []
paired_indices = []
tmp = 0
for key in ntu_action_multi_enumerator:
    caps.extend(ntu_action_multi_enumerator[key])
    paired_indices.append(list(range(tmp, tmp+len(ntu_action_multi_enumerator[key]))))
    tmp+=len(ntu_action_multi_enumerator[key])

cap2key = {caps[i]:i for i in range(len(caps))}

def plot_t2m(data, result_path, caption):
    joint = recover_from_ric(torch.from_numpy(data).float(), opt.joints_num).numpy()
    print((joint[:,:,1].max(axis=1)-joint[:,:,1].min(axis=1)).max(), (joint[:,:,1].max(axis=1)-joint[:,:,1].min(axis=1)).min())
    # joint = motion_temporal_filter(joint, sigma=1)
    plot_3d_motion(result_path, paramUtil.t2m_kinematic_chain, joint, title=caption, fps=20)

def plot_t2m2(data1, data2, result_path, caption):
    joint1, joint2 = recover_from_ric2(torch.from_numpy(data1).unsqueeze(0).float(), torch.from_numpy(data2).unsqueeze(0).float(), 22)
    joint1, joint2 = joint1.squeeze(0).numpy(), joint2.squeeze(0).numpy()
    np.save(result_path.replace('gif','npy'), np.array([joint1, joint2]))
    # joint = motion_temporal_filter(joint, sigma=1)
    plot_3d_motion2(result_path, paramUtil.t2m_kinematic_chain, joint1, joint2, title=caption, fps=20)
    print(result_path, 'done')


def build_mul_models(opt):
    encoder = MotionInteractionTransformer(
        input_feats=opt.dim_pose,
        num_frames=opt.max_motion_length,
        num_layers=opt.num_layers,
        latent_dim=opt.latent_dim,
        no_clip=opt.no_clip,
        no_eff=opt.no_eff,
        no_cross_attn=opt.no_cross_attn,
        cap_id=opt.cap_id
        )
    return encoder

def build_models(opt):
    encoder = MotionTransformer(
        input_feats=opt.dim_pose,
        num_frames=opt.max_motion_length,
        num_layers=opt.num_layers,
        latent_dim=opt.latent_dim,
        no_clip=opt.no_clip,
        no_eff=opt.no_eff)
    return encoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt_path', type=str, help='Opt path')
    parser.add_argument('--text_category', type=int, default=0, help='')
    parser.add_argument('--motion_length', type=int, default=60, help='Number of frames for motion generation')
    parser.add_argument('--result_path', type=str, default="test_sample.gif", help='Path to save generation result')
    parser.add_argument('--gpu_id', type=int, default=-1, help="which gpu to use")
    parser.add_argument('--which_epoch', type=str, default="latest", help='Checkpoint that will be used')
    parser.add_argument('--interaction', action="store_true", help='')
    parser.add_argument('--cap_id', action="store_true", help='Whether to input interaction category')
    args = parser.parse_args()
    
    device = torch.device('cuda:%d' % args.gpu_id if args.gpu_id != -1 else 'cpu')
    opt = get_opt(args.opt_path, args.which_epoch, device)

    assert args.motion_length <= 196

    dim_word = 300
    dim_pos_ohot = len(POS_enumerator)
    num_classes = 200 // opt.unit_length
    
    mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
    std = np.load(pjoin(opt.meta_dir, 'std.npy'))

    if args.interaction:
        mean, init_mean = mean[:-4], mean[-4:]
        std, init_std = std[:-4], std[-4:]

        encoder = build_mul_models(opt).to(device)
        trainer = DDPMMulTrainer(opt, encoder)
        trainer.load(pjoin(opt.model_dir, opt.which_epoch + '.tar'))

        trainer.eval_mode()
        trainer.to(opt.device)

        result_dict = {}
        pair = paired_indices[args.text_category]
        if args.cap_id:
            if len(pair)==2:
                caption1 = [pair[0]]
                caption2 = [pair[1]]
            else:
                caption1 = [pair[0]]
                caption2 = [pair[0]]
        else:
            if len(pair)==2:
                caption1 = [caps[pair[0]]]#['A person is pushing the other person.']
                caption2 = [caps[pair[1]]]#['A person is pushed by the other person.']
            else:
                caption1 = [caps[pair[0]]]
                caption2 = [caps[pair[0]]]

        print('input:', caption1, caption2)
        res_dir_path = os.path.join(args.result_path, f'{caption1[0]}_{caption2[0]}')
        if os.path.exists(res_dir_path):
            shutil.rmtree(res_dir_path)
        os.makedirs(res_dir_path, exist_ok=True)
        with torch.no_grad():
            if args.motion_length != -1:
                m_lens = torch.LongTensor([args.motion_length]).to(device)
                pred_motions = trainer.generate(caption1, caption2, m_lens, opt.dim_pose)[0]
                motion1, motion2 = pred_motions[0].cpu().numpy(), pred_motions[1].cpu().numpy()

                motion1[1:] = motion1[1:] * std  + mean
                motion2[1:] = motion2[1:] * std  + mean
                motion1[0,:4] = motion1[0,:4] * init_std  + init_mean
                motion2[0,:4] = motion2[0,:4] * init_std  + init_mean

                motion1 = np.concatenate([motion1[1:], motion1[0][None,:]], axis=0)
                motion2 = np.concatenate([motion2[1:], motion2[0][None,:]], axis=0)

                if args.cap_id:
                    title = caps[caption1[0]] + '#' + caps[caption2[0]] + " #%d" % motion1.shape[0]
                else:
                    title = caption1[0] + '#' + caption2[0] + " #%d" % motion1.shape[0]
                plot_t2m2(motion1, motion2, os.path.join(res_dir_path, 'generated.gif'), title)

    else:
        mean_tmp = np.load(pjoin('checkpoints/ntu_mul/t2m_sample/meta', 'mean.npy'))[:263]
        std_tmp = np.load(pjoin('checkpoints/ntu_mul/t2m_sample/meta', 'std.npy'))[:263]

        encoder = build_models(opt).to(device)
        trainer = DDPMTrainer(opt, encoder)
        trainer.load(pjoin(opt.model_dir, opt.which_epoch + '.tar'))

        trainer.eval_mode()
        trainer.to(opt.device)

        result_dict = {}
        for ii in range(15):
            with torch.no_grad():
                if args.motion_length != -1:
                    caption = [args.text]
                    m_lens = torch.LongTensor([args.motion_length]).to(device)
                    pred_motions = trainer.generate(caption, m_lens, opt.dim_pose)
                    motion = pred_motions[0].cpu().numpy()
                    motion, motion_tmp = motion * std + mean, motion * std_tmp + mean_tmp
                    title = args.text + " #%d" % motion.shape[0]
                    
                    plot_t2m(motion, os.path.join(args.result_path, 'one_motion_'+str(ii)+'.gif'), title)
                    plot_t2m(motion_tmp, os.path.join(args.result_path, 'one_motion_tmp_'+str(ii)+'.gif'), title)

import argparse
from datetime import datetime
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
from datasets import get_dataset_motion_loader, get_motion_loader
from models import MotionInteractionTransformer
from utils.get_opt import get_opt
from utils.metrics import *
from datasets import EvaluatorModelWrapper
from collections import OrderedDict
from utils.plot_script import *
from utils import paramUtil
from utils.utils import *
from trainers import DDPMMulTrainer

from os.path import join as pjoin


def build_models(opt, dim_pose):
    encoder = MotionInteractionTransformer(
        input_feats=dim_pose,
        num_frames=opt.max_motion_length,
        num_layers=opt.num_layers,
        latent_dim=opt.latent_dim,
        no_clip=opt.no_clip,
        no_eff=opt.no_eff,
        no_cross_attn=opt.no_cross_attn,
        cap_id=opt.cap_id
        )
    return encoder


torch.multiprocessing.set_sharing_strategy('file_system')


def evaluate_matching_score(motion_loaders, file, save_dir, file_id):

    acc_dict = OrderedDict({})
    consistency_dict = OrderedDict({})
    activation_dict = OrderedDict({})
    activation2_dict = OrderedDict({})

    print('========== Evaluating Matching Score ==========')
    for motion_loader_name, motion_loader in motion_loaders.items():
        print(motion_loader_name)
        all_motion_embeddings = []
        all_fin_motion_embeddings = []
        acc_list = []
        consistency_acc_list = []
        pred_list = []
        gt_list = []
        score_list = []
        all_size = 0
        matching_score_sum = 0
        top_k_count = 0

        with torch.no_grad():
            for idx, batch in enumerate(motion_loader):
                class_id, _, _, motions1, motions2, m_lens = batch

                fin_embeddings, motion_embeddings, consitency_embedding = eval_wrapper.get_motion_embeddings(
                    motions1=motions1,
                    motions2=motions2,
                    m_lens=m_lens
                )
                predicted_classes = fin_embeddings.max(dim=1).indices.cpu()
                consitency_pred = consitency_embedding.max(dim=1).indices.cpu()


                acc_list.extend(predicted_classes.numpy()==class_id.numpy())
                consistency_acc_list.extend(consitency_pred.numpy()==0)
                pred_list.extend(predicted_classes.numpy().tolist())
                gt_list.extend(class_id.numpy().tolist())
                all_motion_embeddings.append(motion_embeddings.cpu().numpy())
                all_fin_motion_embeddings.append(fin_embeddings.cpu().numpy())

            all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
            all_fin_motion_embeddings = np.concatenate(all_fin_motion_embeddings, axis=0)

            activation_dict[motion_loader_name] = all_motion_embeddings
            activation2_dict[motion_loader_name] = all_fin_motion_embeddings

            acc_dict[motion_loader_name] = sum(acc_list)/len(acc_list)
            consistency_dict[motion_loader_name] = sum(consistency_acc_list)/len(consistency_acc_list)

            fig, ax = plt.subplots(figsize=(20, 20))
            cm = confusion_matrix(gt_list, pred_list)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(ax=ax)
            plt.savefig(os.path.join(save_dir, motion_loader_name+'_confusion_matrix{}.png'.format(file_id)))
            plt.close()

    return acc_dict, activation_dict, activation2_dict, consistency_dict


def evaluate_fid(groundtruth_loader, activation_dict, file):
    
    gt_motion_embeddings = []
    gt2_motion_embeddings = []
    print('========== Evaluating FID ==========')
    gt_mu, gt_cov = calculate_activation_statistics(activation_dict['ground truth'])
    
    def calc_one(act_dict, gt_mu, gt_cov):
        eval_dict = OrderedDict({})
        for model_name, motion_embeddings in act_dict.items():
            mu, cov = calculate_activation_statistics(motion_embeddings)
            # print(mu)
            fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
            print(f'---> [{model_name}] FID: {fid:.4f}')
            print(f'---> [{model_name}] FID: {fid:.4f}', file=file, flush=True)
            eval_dict[model_name] = fid
        return eval_dict
    eval_dict = calc_one(activation_dict, gt_mu, gt_cov)

    return eval_dict


def evaluate_diversity(activation_dict, file):
    eval_dict = OrderedDict({})
    print('========== Evaluating Diversity ==========')
    for model_name, motion_embeddings in activation_dict.items():
        diversity = calculate_diversity(motion_embeddings, diversity_times)
        eval_dict[model_name] = diversity
        print(f'---> [{model_name}] Diversity: {diversity:.4f}')
        print(f'---> [{model_name}] Diversity: {diversity:.4f}', file=file, flush=True)
    return eval_dict


def evaluate_multimodality(mm_motion_loaders, file):
    eval_dict = OrderedDict({})
    print('========== Evaluating MultiModality ==========')
    for model_name, mm_motion_loader in mm_motion_loaders.items():
        print(model_name)
        mm_motion_embeddings = []
        with torch.no_grad():
            for idx, batch in enumerate(mm_motion_loader):
                # (1, mm_replications, dim_pos)
                motions1, motions2, m_lens = batch
                _, motion_embeddings, _ = eval_wrapper.get_motion_embeddings(motions1[0], motions2[0], m_lens[0])
                mm_motion_embeddings.append(motion_embeddings.unsqueeze(0))
        if len(mm_motion_embeddings) == 0:
            multimodality = 0
        else:
            mm_motion_embeddings = torch.cat(mm_motion_embeddings, dim=0).cpu().numpy()
            multimodality = calculate_multimodality(mm_motion_embeddings, mm_num_times)
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}')
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}', file=file, flush=True)
        eval_dict[model_name] = multimodality
    return eval_dict


def get_metric_statistics(values):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def evaluation(log_file, save_dir, file_id):
    with open(log_file, 'w') as f:
        all_metrics = OrderedDict({'Acc': OrderedDict({}),
                                   'FID': OrderedDict({}),
                                   'FID2': OrderedDict({}),
                                   'Consistency': OrderedDict({}),
                                   'Diversity': OrderedDict({}),
                                   'MultiModality': OrderedDict({})})
        for replication in range(replication_times):
            motion_loaders = {}
            mm_motion_loaders = {}
            motion_loaders['ground truth'] = gt_loader
            for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():
                motion_loader, mm_motion_loader, gt_mm_motion_loader = motion_loader_getter()
                motion_loaders[motion_loader_name] = motion_loader
                mm_motion_loaders[motion_loader_name] = mm_motion_loader
                mm_motion_loaders['ground truth'] = gt_mm_motion_loader
            
            print(f'==================== Replication {replication} ====================')
            print(f'==================== Replication {replication} ====================', file=f, flush=True)
            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            acc_dict, acti_dict, acti2_dict, consistency_dict = evaluate_matching_score(motion_loaders, f, save_dir, file_id)
            
            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            fid_score_dict = evaluate_fid(gt_loader, acti_dict, f)
            
            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            div_score_dict = evaluate_diversity(acti_dict, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            mm_score_dict = evaluate_multimodality(mm_motion_loaders, f)

            print(f'!!! DONE !!!')
            print(f'!!! DONE !!!', file=f, flush=True)
            for key, item in acc_dict.items():
                if key not in all_metrics['Acc']:
                    all_metrics['Acc'][key] = [item]
                else:
                    all_metrics['Acc'][key] += [item]

            for key, item in consistency_dict.items():
                if key not in all_metrics['Consistency']:
                    all_metrics['Consistency'][key] = [item]
                else:
                    all_metrics['Consistency'][key] += [item]
            
            for key, item in fid_score_dict.items():
                if key not in all_metrics['FID']:
                    all_metrics['FID'][key] = [item]
                else:
                    all_metrics['FID'][key] += [item]

            for key, item in div_score_dict.items():
                if key not in all_metrics['Diversity']:
                    all_metrics['Diversity'][key] = [item]
                else:
                    all_metrics['Diversity'][key] += [item]

            for key, item in mm_score_dict.items():
                if key not in all_metrics['MultiModality']:
                    all_metrics['MultiModality'][key] = [item]
                else:
                    all_metrics['MultiModality'][key] += [item]

        # print(all_metrics['Diversity'])
        for metric_name, metric_dict in all_metrics.items():
            print('========== %s Summary ==========' % metric_name)
            print('========== %s Summary ==========' % metric_name, file=f, flush=True)

            for model_name, values in metric_dict.items():
                # print(metric_name, model_name)
                mean, conf_interval = get_metric_statistics(np.array(values))
                # print(mean, mean.dtype)
                if isinstance(mean, np.float64) or isinstance(mean, np.float32):
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}')
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}', file=f, flush=True)
                elif isinstance(mean, np.ndarray):
                    line = f'---> [{model_name}]'
                    for i in range(len(mean)):
                        line += '(top %d) Mean: %.4f CInt: %.4f;' % (i+1, mean[i], conf_interval[i])
                    print(line)
                    print(line, file=f, flush=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt_path', type=str, help='Opt path')
    parser.add_argument('--model_name', type=str, default='latest', help='')
    parser.add_argument('--gpu_id', type=int, default=-1, help="which gpu to use")
    parser.add_argument('--split_file', type=str, default="test_sub.txt", help='Checkpoint that will be used')
    parser.add_argument('--file_id', type=str, default=0, help="which gpu to use")
    args = parser.parse_args()

    #mm_num_samples = 100
    mm_num_repeats = 20
    mm_num_times = 15

    diversity_times = 300
    replication_times = 1
    batch_size = 32
    opt_path = args.opt_path
    model_name = args.model_name
    split_file = args.split_file
    dataset_opt_path = opt_path
    file_id = args.file_id
    
    device_id = args.gpu_id
    device = torch.device('cuda:%d' % device_id if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device_id)

    gt_loader, gt_dataset = get_dataset_motion_loader(dataset_opt_path, batch_size, device, split_file)
    wrapper_opt = get_opt(dataset_opt_path, model_name, device)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    opt = get_opt(opt_path, model_name, device)
    encoder = build_models(opt, opt.dim_pose)
    trainer = DDPMMulTrainer(opt, encoder)
    eval_motion_loaders = {
        'text2motion': lambda: get_motion_loader(
            opt,
            batch_size,
            trainer,
            gt_dataset,
            mm_num_repeats
        )
    }
    
    save_dir = './result/'+opt_path.split('/')[2]+'/'+model_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    log_file = os.path.join(save_dir, 't2m_fin_evaluation{}.log'.format(file_id))
    evaluation(log_file, save_dir, file_id)

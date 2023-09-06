import argparse
import os
import torch
from mmcv.runner import get_dist_info
import torch.distributed as dist


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--name', type=str, default="test", help='Name of this trial')
        self.parser.add_argument('--decomp_name', type=str, default="Decomp_SP001_SM001_H512", help='Name of autoencoder model')

        self.parser.add_argument('--multi', action='store_true', help='whether to use multi person mode')
        self.parser.add_argument('--cap_id', action='store_true', help='caption or id')
        self.parser.add_argument('--causal', action='store_true', help='causal attention or not')
        self.parser.add_argument('--cap_same', action='store_true', help='only input active')
        self.parser.add_argument('--single_transformer', action='store_true', help='use baseline model')
        self.parser.add_argument('--pretrained', action='store_true', help='use pretrained model')
        self.parser.add_argument('--only_language', action='store_true', help='use pretrained model (only language)')
        self.parser.add_argument('--only_motion', action='store_true', help='use pretrained model (only motion)')
        self.parser.add_argument('--label_path', type=str, help='passive/active annotation path')
        self.parser.add_argument('--save_label_dir', type=str, help='passive/active annotation path')

        self.parser.add_argument("--gpu_id", type=int, default=-1, help='GPU id')
        self.parser.add_argument('--port', type=str, default='12345', help='port number')
        self.parser.add_argument("--distributed", action="store_true", help='Weather to use DDP training')

        self.parser.add_argument('--dataset_name', type=str, default='t2m', help='Dataset Name')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')

        self.parser.add_argument("--unit_length", type=int, default=4, help="Motions are cropped to the maximum times of unit_length")
        self.parser.add_argument("--max_text_len", type=int, default=20, help="Maximum length of text description")

        self.parser.add_argument('--text_enc_mod', type=str, default='bigru')
        self.parser.add_argument('--estimator_mod', type=str, default='bigru')

        self.parser.add_argument('--dim_text_hidden', type=int, default=512, help='Dimension of hidden unit in text encoder')
        self.parser.add_argument('--dim_att_vec', type=int, default=512, help='Dimension of attention vector')
        self.parser.add_argument('--dim_z', type=int, default=128, help='Dimension of latent Gaussian vector')

        self.parser.add_argument('--n_layers_pri', type=int, default=1, help='Number of layers in prior network')
        self.parser.add_argument('--n_layers_pos', type=int, default=1, help='Number of layers in posterior network')
        self.parser.add_argument('--n_layers_dec', type=int, default=1, help='Number of layers in generator')

        self.parser.add_argument('--dim_pri_hidden', type=int, default=1024, help='Dimension of hidden unit in prior network')
        self.parser.add_argument('--dim_pos_hidden', type=int, default=1024, help='Dimension of hidden unit in posterior network')
        self.parser.add_argument('--dim_dec_hidden', type=int, default=1024, help='Dimension of hidden unit in generator')

        self.parser.add_argument('--dim_movement_enc_hidden', type=int, default=512,
                                 help='Dimension of hidden in AutoEncoder(encoder)')
        self.parser.add_argument('--dim_movement_dec_hidden', type=int, default=512,
                                 help='Dimension of hidden in AutoEncoder(decoder)')
        self.parser.add_argument('--dim_movement_latent', type=int, default=512, help='Dimension of motion snippet')

        self.initialized = True



    def parse(self):
        if not self.initialized:
            self.initialize()

        self.opt = self.parser.parse_args()

        self.opt.is_train = self.is_train

        if self.opt.gpu_id != -1:
            # self.opt.gpu_id = int(self.opt.gpu_id)
            torch.cuda.set_device(self.opt.gpu_id)

        args = vars(self.opt)

        if args["distributed"]:
            init_dist('slurm')
        rank, world_size = get_dist_info()
        if rank == 0:
            print('------------ Options -------------')
            for k, v in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ----------------')
            if self.is_train:
                # save to the disk
                expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.dataset_name, self.opt.name)
                if not os.path.exists(expr_dir):
                    os.makedirs(expr_dir)
                file_name = os.path.join(expr_dir, 'opt.txt')
                with open(file_name, 'wt') as opt_file:
                    opt_file.write('------------ Options -------------\n')
                    for k, v in sorted(args.items()):
                        opt_file.write('%s: %s\n' % (str(k), str(v)))
                    opt_file.write('-------------- End ----------------\n')
        if world_size > 1:
            dist.barrier()
        return self.opt

# -*- coding: utf-8 -*-

import time
import argparse
import copy
import os
import random
import time
from itertools import product

import numpy as np
import open3d as o3d
import torch
from pyhocon import ConfigFactory
from tqdm import tqdm
from models.NeuralUDF.network import NeuralUDF
from models.NeuralUDF.dataset_NormalizeSpace import Dataset

from extensions.chamfer_dist import ChamferDistance
import torch.nn as nn
from models.NeuralUDF.utils import get_root_logger, print_log
import torch.nn.functional as F
import math
# from models.NeuralUDF.DualMeshUDF import extract_mesh_from_udf
# from DualMeshUDF import write_obj

device="cuda" if torch.cuda.is_available() else "cpu"

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name())





def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)  # Python random module
    np.random.seed(seed_value)  # Numpy library
    torch.manual_seed(seed_value)  # Torch

    # if using CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # if using multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

################### Bone_reconstruction_UDF Implementation ##################################3
class Runner_UDF:
    def __init__(self,dataset_processed,udf_network,base_exp_dir):
        self.device = torch.device('cuda')

        self.dataset_processed = dataset_processed

        self.iter_step = 0
        # loss_NC enforces consistency between the predicted UDF value and the
        # signed projection of a point onto its nearest surface sample.
        self.lambda_NC=1

        # loss_GS supervises the field on a sparse grid with precomputed UDF
        # values so the network does not only fit the observed surface points.
        self.lambda_GS=0.01

        # Training parameters are fixed here rather than in a separate config.
        self.maxiter = 30000
        self.save_freq = 5000
        self.report_freq = 1000
        self.val_freq = 5000
        self.batch_size = 5000
        self.learning_rate = 0.001
        self.warm_up_end = 1000

        # ChamferDistance is used only to identify the closest target point for
        # each projected sample; the returned distances are not used directly.
        self.ChamferDis = ChamferDistance().cuda()
        self.L2_loss=nn.MSELoss()

        self.udf_network = udf_network
        self.udf_optimizer = torch.optim.Adam(self.udf_network.parameters(), lr=self.learning_rate)
        self.base_exp_dir=base_exp_dir
        os.makedirs(base_exp_dir,exist_ok=True)



    def train(self):
        timestamp_start = time.strftime('%Y%m%d_%H%M%S', time.localtime())

        print(f"Start time:{timestamp_start}")
        log_file = os.path.join(os.path.join(self.base_exp_dir), 'logger.log')

        logger = get_root_logger(log_file=log_file, name='outs')
        self.logger = logger

        res_step = self.maxiter - self.iter_step

        grid_sparse=self.dataset_processed.grid_sparse
        grid_sparse_udf_gt=self.dataset_processed.grid_sparse_udf_gt

        # Very small target UDF values are ignored here; the effective
        # supervision grid focuses on regions with meaningful distance signal.
        idx=grid_sparse_udf_gt>0.1
        grid_sparse_udf_gt=grid_sparse_udf_gt[idx]
        grid_sparse=grid_sparse[idx]

        for iter_i in tqdm(range(res_step)):
            self.update_learning_rate_np(iter_i)

            samples,samples_near, samples_near_normal,pcd_gt,pcd_gt_normals = self.dataset_processed.np_train_data(self.batch_size)

            ###########Train Bone_reconstruction_UDF Network################
            self.udf_optimizer.zero_grad()
            samples.requires_grad = True
            # The UDF gradient approximates the surface normal direction.
            gradients_sample = self.udf_network.gradient(samples).squeeze()  # 5000x3

            udf_sample = self.udf_network.udf(samples)  # 5000x1
            grad_norm = F.normalize(gradients_sample, dim=1)

            # Project each sample toward the implicit surface by moving along
            # the normalized UDF gradient by the predicted distance magnitude.
            samples_moved = samples - grad_norm * udf_sample
            _, idx1, idx2 = self.ChamferDis(samples_moved.unsqueeze(0), pcd_gt.unsqueeze(0))

            # Near-surface consistency:
            # The dot product measures how far the sample is from the closest
            # ground-truth point along the estimated normal direction. This
            # should match the network's predicted unsigned distance.
            loss_NC = torch.abs(
                (grad_norm * (samples - pcd_gt[idx1[0]])).sum(dim=1,
                                                              keepdim=True) - udf_sample).mean()

            # Sparse-grid supervision anchors the field away from the surface.
            grid_sparse_udf_pred = self.udf_network.udf(grid_sparse)
            grid_sprase_loss = self.L2_loss(grid_sparse_udf_pred, grid_sparse_udf_gt)
            total_loss = self.lambda_NC * loss_NC + self.lambda_GS * grid_sprase_loss

            total_loss.backward()
            self.udf_optimizer.step()


            ############# Saving #################
            self.iter_step += 1
            if self.iter_step % self.report_freq == 0:
                print_log('iter: {:8>d} total_loss = {} lr = {}'.format(self.iter_step, total_loss,
                                                                        self.udf_optimizer.param_groups[0]['lr']),
                          logger=logger)

            if self.iter_step %self.save_freq==0 and self.iter_step>=15000*0:

                self.save_checkpoint()






    def update_learning_rate_np(self, iter_step):
        # Linear warm-up followed by cosine decay.
        warn_up = self.warm_up_end
        max_iter = self.maxiter
        init_lr = self.learning_rate
        lr = (iter_step / warn_up) if iter_step < warn_up else 0.5 * (
                    math.cos((iter_step - warn_up) / (max_iter - warn_up) * math.pi) + 1)
        lr = lr * init_lr
        for g in self.udf_optimizer.param_groups:
            g['lr'] = lr




    def load_checkpoint(self, checkpoint_name):
        checkpoint_file=os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name)
        print(checkpoint_file)
        assert os.path.isfile(checkpoint_file),f"file does not exist:{checkpoint_file}"

        checkpoint = torch.load(checkpoint_file, map_location=self.device)

        self.udf_network.load_state_dict(checkpoint['udf_network_fine'])

        self.iter_step = checkpoint['iter_step']

    def save_checkpoint(self):
        # Only the network weights and the current iteration are needed to
        # resume training from this script.
        checkpoint = {
            'udf_network_fine': self.udf_network.state_dict(),
            'iter_step': self.iter_step,
        }
        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint,
                   os.path.join(self.base_exp_dir, 'checkpoints', f"ckpt_{self.iter_step:0>6d}_NeuralUDF.pth"))





def train_NeuralUDF(dataset_processed,udf_network,base_exp_dir):
    # Thin wrapper used by registration_core.load_udf_network when a checkpoint is missing.
    runner = Runner_UDF(dataset_processed,udf_network,base_exp_dir)
    runner.train()










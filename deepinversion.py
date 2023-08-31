# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Official PyTorch implementation of CVPR2020 paper
# Dreaming to Distill: Data-free Knowledge Transfer via DeepInversion
# Hongxu Yin, Pavlo Molchanov, Zhizhong Li, Jose M. Alvarez, Arun Mallya, Derek
# Hoiem, Niraj K. Jha, and Jan Kautz
# --------------------------------------------------------

##########################################################################
# File adapted from: https://github.com/NVlabs/DeepInversion/tree/master #
##########################################################################

import torch
import torch.nn as nn
import torch.optim as optim
import collections
import torch.cuda.amp as amp
import random
import torch
import torchvision.utils as vutils
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import pdb
import time
import warnings
import psutil
import lmdb
import io
import pickle5 as pickle


def create_folder(directory):
    # from https://stackoverflow.com/a/273227
    if not os.path.exists(directory):
        os.makedirs(directory)


def denormalize(image_tensor):
    '''
    convert floats back to input
    '''
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c] * s + m, 0, 1)

    return image_tensor


def clip(image_tensor):
    '''
    adjust the input based on mean and variance
    '''
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m / s, (1 - m) / s)
    return image_tensor


def lr_policy(lr_fn):
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return _alr


def lr_cosine_policy(base_lr, warmup_length, epochs):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn)


class DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        #forcing mean and variance to match between two distributions
        #other ways might work better, i.g. KL divergence
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2)

        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()


def get_image_prior_losses(inputs_jit):
    # COMPUTE total variation regularization loss
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + (
            diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
    loss_var_l1 = loss_var_l1 * 255.0
    return loss_var_l1, loss_var_l2


class DeepInversionClass(object):
    def __init__(self, bs=84, num_batches=1,
                 use_fp16=True, net_teacher=None, path="./runs/",
                 parameters=dict(),
                 setting_id=0,
                 jitter=30,
                 criterion=None,
                 coefficients=dict(),
                 network_output_function=lambda x: x,
                 hook_for_display = None, 
                 num_classes=10,
                 device='cuda:0', 
                 task=None, 
                 dataset=None):
        '''
        :param bs: batch size per GPU for image generation
        :param use_fp16: use FP16 (or APEX AMP) for model inversion, uses less memory and is faster for GPUs with Tensor Cores
        :parameter net_teacher: Pytorch model to be inverted
        :param path: path where to write temporal images and data
        :param final_data_path: path to write final images into
        :param parameters: a dictionary of control parameters:
            "resolution": input image resolution, single value, assumed to be a square, 224
            "random_label" : for classification initialize target to be random values
            "start_noise" : start from noise, def True, other options are not supported at this time
            "detach_student": if computing Adaptive DI, should we detach student?
        :param setting_id: predefined settings for optimization:
            0 - will run low resolution optimization for 1k and then full resolution for 1k;
            1 - will run optimization on high resolution for 2k
            2 - will run optimization on high resolution for 20k

        :param jitter: amount of random shift applied to image at every iteration
        :param coefficients: dictionary with parameters and coefficients for optimization.
            keys:
            "r_feature" - coefficient for feature distribution regularization
            "tv_l1" - coefficient for total variation L1 loss
            "tv_l2" - coefficient for total variation L2 loss
            "l2" - l2 penalization weight
            "lr" - learning rate for optimization
            "main_loss_multiplier" - coefficient for the main loss optimization
            "adi_scale" - coefficient for Adaptive DeepInversion, competition, def =0 means no competition
        network_output_function: function to be applied to the output of the network to get the output
        hook_for_display: function to be executed at every print/save call, useful to check accuracy of verifier
        '''

        print("Deep inversion class generation")
        # for reproducibility
        torch.manual_seed(torch.cuda.current_device())

        self.debug_output = False

        self.net_teacher = net_teacher
        self.net_teacher.eval()

        if "resolution" in parameters.keys():
            self.image_resolution = parameters["resolution"]
            self.random_label = parameters["random_label"]
            self.start_noise = parameters["start_noise"]
            self.detach_student = parameters["detach_student"]
            self.do_flip = parameters["do_flip"]
            self.store_best_images = parameters["store_best_images"]
        else:
            self.image_resolution = 224
            self.random_label = False
            self.start_noise = True
            self.detach_student = False
            self.do_flip = True
            self.store_best_images = False

        self.setting_id = setting_id
        self.bs = bs  # batch size
        self.num_batches = num_batches
        self.use_fp16 = use_fp16
        self.save_every = 100
        self.jitter = jitter
        self.criterion = criterion
        self.network_output_function = network_output_function
        do_clip = True
        self.first_class = 0 # first class which should be generated in a new batch
        self.num_classes = num_classes
        self.device = device
        if task is not None:
            self.task = task
        else:
            raise ValueError('No task defined in the Deep Inversion Class.')

        if self.task == 'detection':
            self.img_metas = {'img_shape': (3, self.image_resolution, self.image_resolution), 'scale_factor': 1, 
                            'flip': False, 'filename': '', 'orig_shape': (3, self.image_resolution, self.image_resolution), 
                            'pad_shape': (3, self.image_resolution, self.image_resolution), 
                            'img_norm_cfg': {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2471, 0.2435, 0.2616], 'to_rgb': False}}

        if dataset is not None:
            self.dataset = dataset
        else:
            raise ValueError('No dataset defined in the Deep Inversion Class.')

        self.use_lmdb = False

        if "r_feature" in coefficients:
            self.bn_reg_scale = coefficients["r_feature"]
            self.first_bn_multiplier = coefficients["first_bn_multiplier"]
            self.var_scale_l1 = coefficients["tv_l1"]
            self.var_scale_l2 = coefficients["tv_l2"]
            self.l2_scale = coefficients["l2"]
            self.lr = coefficients["lr"]
            self.main_loss_multiplier = coefficients["main_loss_multiplier"]
            self.adi_scale = coefficients["adi_scale"]
            self.betas = coefficients["betas"]
        else:
            print("Provide a dictionary with the weights of the different loss parts.")

        self.num_generations = 0

        ## Create folders for images and logs
        prefix = path
        self.prefix = prefix
        self.final_data_path = prefix + 'images/'
        self.res_file = self.prefix + 'results_DI.txt'
        with open(self.res_file, 'a') as f:
            f.write('Use the Deep Inversion method to generate pseudo images.' + '\n')

        local_rank = torch.cuda.current_device()
        if local_rank==0 and self.debug_output:
            create_folder(prefix + "/debug_output_images/")
        if self.final_data_path is not None:
            create_folder(self.final_data_path)

        ## Create hooks for feature statistics
        self.loss_r_feature_layers = []

        for module in self.net_teacher.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.loss_r_feature_layers.append(DeepInversionFeatureHook(module))

        self.hook_for_display = None
        if hook_for_display is not None:
            self.hook_for_display = hook_for_display


    @staticmethod
    def get_hyperparameters(opt):
        # high resolution
        # Pose
        if opt.dataset in ['COCO'] and 'resnet' in opt.model and opt.task in ['pose']:
            setting_id = 0
            jitter = 30

            if opt.img_size < 100:
                warnings.warn('WARNING! You use the high resolution dataset with a image size below 100 px.')

            parameters = dict()
            parameters["resolution"] = opt.img_size
            parameters["random_label"] = False
            parameters["start_noise"] = True
            parameters["detach_student"] = False
            parameters["do_flip"] = True
            parameters["store_best_images"] = False

            coefficients = dict()
            coefficients["r_feature"] = 0.01
            coefficients["first_bn_multiplier"] = 10.0
            coefficients["tv_l1"] = 0.0
            coefficients["tv_l2"] = 0.0001
            coefficients["l2"] = 1e-5
            coefficients["lr"] = 0.25
            coefficients["betas"] = [0.5, 0.9] if setting_id in [0,1] else [0.9, 0.999]
            coefficients["main_loss_multiplier"] = 1.0
            coefficients["adi_scale"] = 0.0

        # detection
        elif opt.dataset in ['COCO', 'VOC'] and ('resnet' in opt.model  or 'vgg' in opt.model) and opt.task in ['detection']:
            setting_id = 0
            jitter = 30

            if opt.img_size < 100:
                warnings.warn('WARNING! You use the high resolution dataset with a image size below 100 px.')
            print('Use the detection DI hyperparameters.')
            parameters = dict()
            parameters["resolution"] = opt.img_size
            parameters["random_label"] = False
            parameters["start_noise"] = True
            parameters["detach_student"] = False
            parameters["do_flip"] = True
            parameters["store_best_images"] = False

            coefficients = dict()
            # no real structures/objects are optimized. Maybe some pseudo labels are needed for good structures/objects
            coefficients["r_feature"] = 0.01 # makes the structures
            coefficients["first_bn_multiplier"] = 10.0
            coefficients["tv_l1"] = 0.0
            coefficients["tv_l2"] = 0.0001 #makes it more colorful
            coefficients["l2"] = 1e-5 # makes it more smooth
            coefficients["lr"] = 0.25
            coefficients["betas"] = [0.5, 0.9] if setting_id in [0,1] else [0.9, 0.999]
            coefficients["main_loss_multiplier"] = 1.0
            coefficients["adi_scale"] = 0.0

        # low resolution
        # CIFAR classification
        elif opt.dataset in ['CIFAR10', 'CIFAR100']:
            setting_id = 1
            jitter = 2

            parameters = dict()
            parameters["resolution"] = opt.img_size
            parameters["random_label"] = False
            parameters["start_noise"] = True
            parameters["detach_student"] = False
            parameters["do_flip"] = False
            parameters["store_best_images"] = False

            coefficients = dict()
            coefficients["r_feature"] = 10.0
            coefficients["first_bn_multiplier"] = 1.0
            coefficients["tv_l1"] = 0.0
            coefficients["tv_l2"] = 0.001
            coefficients["l2"] = 0.0
            coefficients["lr"] = 0.1
            coefficients["betas"] = [0.9, 0.99]
            coefficients["main_loss_multiplier"] = 1.0
            coefficients["adi_scale"] = 0.0

        else:
            raise ValueError('Unknown dataset and model combination in the DI hyperparameter selection. Dataset: %s, Model: %s'%(opt.dataset, opt.model))

        return setting_id, jitter, parameters, coefficients


    def get_batch(self, iter, net_student=None, targets=None):
        # print("get_images call")

        net_teacher = self.net_teacher
        save_every = self.save_every

        kl_loss = nn.KLDivLoss(reduction='batchmean').to(self.device)
        local_rank = torch.cuda.current_device()
        criterion = self.criterion

        best_loss = 1e6
        best_loss_ce = 0
        best_loss_var_l1 = 0
        best_loss_var_l2 = 0
        best_loss_distr = 0
        best_loss_verifier_cig = 0
        best_loss_l2 = 0

        # setup target labels
        if targets is None and self.task == 'classification':
            #only works for classification now, for other tasks need to provide target vector
            if self.random_label:
                targets = torch.LongTensor([random.randint(0, self.num_classes-1) for _ in range(self.bs)]).to(self.device)
            else:
                # Generate the pseudo labels
                if self.num_classes > self.bs:
                    if self.first_class + self.bs < self.num_classes:
                        targets = [i for i in range(self.first_class, self.first_class + self.bs)]
                        self.first_class = self.first_class + self.bs
                    else:
                        new = self.first_class + self.bs - self.num_classes
                        targets = [i for i in range(self.first_class, self.num_classes)] + [i for i in range(new)]
                        self.first_class = new
                else:
                    b = self.bs // self.num_classes
                    r = self.bs % self.num_classes
                    if self.first_class + r < self.num_classes: # generate remaining classes with no new start
                        res = [i for i in range(self.first_class, self.first_class + r)]
                        self.first_class = self.first_class + r
                    else: # generate remaining labels when first_class starts from beginning
                        new = self.first_class + r - self.num_classes
                        res = [i for i in range(self.first_class, self.num_classes)] + [i for i in range(new)]
                        self.first_class = new
                    # All pseudo labels
                    targets = [i for i in range(self.num_classes)] * b + res

                targets = torch.LongTensor(targets).to(self.device)

        else:
            # -1 class for non classification tasks
            targets = torch.LongTensor([-1 for _ in range(self.bs)]).to(self.device)

        inputs = torch.randn((self.bs, 3, self.image_resolution, self.image_resolution), requires_grad=True, device=self.device)
        if self.task == 'detection':
            metas = [self.img_metas] * self.bs
        pooling_function = nn.modules.pooling.AvgPool2d(kernel_size=2)

        if self.setting_id==0:
            skipfirst = False
        else:
            skipfirst = True

        iteration = 0
        for lr_it, lower_res in enumerate([2, 1]):
            if lr_it==0:
                iterations_per_layer = 2000
            else:
                iterations_per_layer = 1000 if not skipfirst else 2000
                if self.setting_id == 2:
                    iterations_per_layer = 20000

            if lr_it==0 and skipfirst:
                continue

            lim_0, lim_1 = self.jitter // lower_res, self.jitter // lower_res

            if self.setting_id == 0:
                #multi resolution, 2k iterations with low resolution, 1k at normal, ResNet50v1.5 works the best, ResNet50 is ok
                optimizer = optim.Adam([inputs], lr=self.lr, betas=self.betas, eps = 1e-8)
                do_clip = True
            elif self.setting_id == 1:
                #2k normal resolultion, for ResNet50v1.5; Resnet50 works as well
                optimizer = optim.Adam([inputs], lr=self.lr, betas=self.betas, eps = 1e-8)
                do_clip = True
            elif self.setting_id == 2:
                #20k normal resolution the closes to the paper experiments for ResNet50
                optimizer = optim.Adam([inputs], lr=self.lr, betas=self.betas, eps = 1e-8)
                do_clip = False

            if self.use_fp16:
                scaler = amp.GradScaler()

            if self.dataset == 'CIFAR10':
                lr_scheduler = lr_cosine_policy(self.lr, 100, iterations_per_layer)

            for iteration_loc in range(iterations_per_layer):
                iteration += 1
                # learning rate scheduling
                if self.dataset == 'CIFAR10':
                    lr_scheduler(optimizer, iteration_loc, iteration_loc)

                # perform downsampling if needed
                if lower_res!=1:
                    inputs_jit = pooling_function(inputs)
                else:
                    inputs_jit = inputs

                # apply random jitter offsets
                off1 = random.randint(-lim_0, lim_0)
                off2 = random.randint(-lim_1, lim_1)
                inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2, 3))

                # Flipping
                flip = random.random() > 0.5
                if flip and self.do_flip:
                    inputs_jit = torch.flip(inputs_jit, dims=(3,))

                # forward pass
                optimizer.zero_grad()
                net_teacher.zero_grad()

                with amp.autocast(enabled=self.use_fp16):
                    outputs = net_teacher(inputs_jit)
                    outputs = self.network_output_function(outputs)

                    # R_cross classification loss
                    if self.task == 'classification':
                        ce_loss = criterion(outputs, targets)
                        loss = ce_loss
                    else:
                        ce_loss = torch.zeros(1, device=self.device)
                        loss = torch.zeros(1, device=self.device)

                    # R_prior losses
                    loss_var_l1, loss_var_l2 = get_image_prior_losses(inputs_jit)

                    # R_feature loss
                    rescale = [self.first_bn_multiplier] + [1. for _ in range(len(self.loss_r_feature_layers)-1)]
                    loss_r_feature = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(self.loss_r_feature_layers) if mod.r_feature < 10**7]) 

                    # R_ADI
                    loss_verifier_cig = torch.zeros(1)
                    if self.adi_scale!=0.0:
                        if self.detach_student:
                            outputs_student = net_student(inputs_jit).detach()
                        else:
                            outputs_student = net_student(inputs_jit)

                        T = 3.0
                        if 1:
                            T = 3.0
                            # Jensen Shanon divergence:
                            # another way to force KL between negative probabilities
                            P = nn.functional.softmax(outputs_student / T, dim=1)
                            Q = nn.functional.softmax(outputs / T, dim=1)
                            M = 0.5 * (P + Q)

                            P = torch.clamp(P, 0.01, 0.99)
                            Q = torch.clamp(Q, 0.01, 0.99)
                            M = torch.clamp(M, 0.01, 0.99)
                            eps = 0.0
                            loss_verifier_cig = 0.5 * kl_loss(torch.log(P + eps), M) + 0.5 * kl_loss(torch.log(Q + eps), M)
                            # JS criteria - 0 means full correlation, 1 - means completely different
                            loss_verifier_cig = 1.0 - torch.clamp(loss_verifier_cig, 0.0, 1.0)

                        if local_rank==0:
                            if iteration % save_every==0:
                                print('loss_verifier_cig', loss_verifier_cig.item())

                    # l2 loss on images
                    if self.dataset == 'CIFAR10':
                        loss_l2 = torch.norm(inputs_jit, 2)
                    else:
                        loss_l2 = torch.norm(inputs_jit.view(self.bs, -1), dim=1).mean()

                    # combining losses
                    loss_aux = self.var_scale_l2 * loss_var_l2 + \
                            self.var_scale_l1 * loss_var_l1 + \
                            self.bn_reg_scale * loss_r_feature + \
                            self.l2_scale * loss_l2

                    # if iteration %100 == 0:
                    #     print('var_l2: %f, var_l1: %f, bn: %f, l2: %f, main: %f'%(loss_var_l2.item(), loss_var_l1.item(), loss_r_feature.item(), loss_l2.item(), loss.item()))

                    if self.adi_scale!=0.0:
                        loss_aux += self.adi_scale * loss_verifier_cig

                    loss = self.main_loss_multiplier * loss + loss_aux

                    # if iteration_loc % 100 == 0:
                    #     print('bn loss: ', [mod.r_feature.item() for (idx, mod) in enumerate(self.loss_r_feature_layers)])
                    #     print('loss_var_l2: ', loss_var_l2.item())
                    #     print('loss_var_l1: ', loss_var_l1.item())
                    #     print('loss_r_feature: ', loss_r_feature.item())
                    #     print('loss_l2: ', loss_l2.item())

                if local_rank==0 and self.debug_output:
                    if iteration % save_every==0:
                        print("------------iteration {}----------".format(iteration))
                        print("total loss", loss.item())
                        print("loss_r_feature", loss_r_feature.item())
                        if self.task == 'classification':
                            print("main criterion", criterion(outputs, targets).item())

                        if self.hook_for_display is not None:
                            self.hook_for_display(inputs, targets)

                # do image update
                if self.use_fp16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    # # optimizer.backward(loss)
                    # with amp.scale_loss(loss, optimizer) as scaled_loss:
                    #     scaled_loss.backward()
                else:
                    loss.backward()
                    optimizer.step()

                # clip color outlayers
                if do_clip:
                    inputs.data = clip(inputs.data)

                if best_loss > loss.item() or iteration == 1:
                    # best_inputs = inputs.data.clone()
                    # best_loss = loss.item()
                    best_loss = loss.item()
                    best_inputs = inputs.clone()
                    best_loss_ce = ce_loss.item()
                    best_loss_var_l1 = loss_var_l1.item()
                    best_loss_var_l2 = loss_var_l2.item()
                    best_loss_distr = loss_r_feature.item()
                    best_loss_verifier_cig = loss_verifier_cig.item()
                    best_loss_l2 = loss_l2.item()

                if iteration % save_every==0 and (save_every > 0) and self.debug_output:
                    if local_rank==0:
                        vutils.save_image(inputs,
                                          '{}debug_output_images/output_{:05d}_gpu_{}.png'.format(self.prefix,
                                                                                           iteration // save_every,
                                                                                           local_rank),
                                          normalize=True, scale_each=True, nrow=int(10))

        if self.store_best_images:
            best_inputs = denormalize(best_inputs)
            self.save_images(best_inputs, targets)

        out = 'Iteration %i: Overall loss: %f; CE loss: %f; var_l1 loss: %f; var_l2 loss: %f; distr loss: %f; verifier_cig loss: %f; L2 loss: %f'%(
                            iter, best_loss, best_loss_ce, best_loss_var_l1, best_loss_var_l2, best_loss_distr, best_loss_verifier_cig, best_loss_l2)

        with open(self.res_file, 'a') as f:
            f.write(out + '\n')

        # to reduce memory consumption by states of the optimizer we deallocate memory
        optimizer.state = collections.defaultdict(dict)

        return best_inputs.detach().cpu(), targets.detach().cpu()


    def save_images(self, images, targets):
        # method to store generated images locally
        local_rank = torch.cuda.current_device()
        for id in range(images.shape[0]):
            class_id = targets[id].item()
            if 0:
                #save into separate folders
                place_to_store = '{}/s{:03d}/img_{:05d}_id{:03d}_gpu_{}_2.jpg'.format(self.final_data_path, class_id,
                                                                                          self.num_generations, id,
                                                                                          local_rank)
            else:
                place_to_store = '{}/img_s{:03d}_{:05d}_id{:03d}_gpu_{}_2.jpg'.format(self.final_data_path, class_id,
                                                                                          self.num_generations, id,
                                                                                          local_rank)

            image_np = images[id].data.cpu().numpy().transpose((1, 2, 0))
            pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
            pil_image.save(place_to_store)


    def generate_pseudo_dataset(self, net_student=None, targets=None, output_path=None):
        # for ADI detach student and add put to eval mode
        start = time.time()
        self.net_teacher.eval()

        # fix net_student
        if not (net_student is None):
            net_student = net_student.eval()
        

        images, targets_out = list(), list()

        for iter in tqdm(range(self.num_batches), total=self.num_batches):
            output, targets_o = self.get_batch(iter, net_student=net_student, targets=targets)
            output = output.detach().cpu()
            targets_o = targets_o.detach().cpu()

            images.append(output)
            targets_out.append(targets_o)

            self.num_generations += 1

            with open(self.res_file, 'a') as f:
                f.write('%f GB of the RAM are used.'%(psutil.virtual_memory().used / 2**30))

        stop = time.time()
        with open(self.res_file, 'a') as f:
            f.write('The generation of %i Deep Inversion batches needed %f seconds.\n'%(self.num_batches, stop - start))
        
        return images, targets_out

import os
import sys
import pdb
import torch
from copy import deepcopy
import torch.nn as nn
import argparse
import yaml
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import pickle5 as pickle
import torchvision
import logging

from loss_functions import get_loss, IntermediateLoss
from pruning_utils import prune_backbone
from CIFAR_utils import CIFAR_test, CIFAR_train
from utils import get_log_dir, PseudoImagesDataset, count_parameters
from pose_utils import POSE_test, POSE_train, POSE_set_hyperparameter, POSE_create_dataset_split
from detection_utils import detection_test, detection_train, detection_create_dataset_split_COCO, detection_create_dataset_split_VOC
from deepinversion import DeepInversionClass
from models.resnet import resnet18, resnet34, resnet50
from models.vgg import vgg16_bn, vgg19_bn


classifiers = {
                'resnet18': resnet18, 'resnet34': resnet34,
                'vgg16': vgg16_bn, 'vgg19': vgg19_bn
                }

pose_models = ['resnet50']

detection_models = ['resnet50', 'vgg16']

# Layers to prune in each stage; first is the conv and second the BN; Name: ModelSize_PruningStages
pruning_stages_vgg = {
    '11':{
        '11_1': [list(range(100))],
        '11_2': [list(range(100))]
    },
    '13':{
        '13_1': [list(range(100))],
        '13_2': [list(range(100))]
    },
    '16':{
        '16_1': [[7, 8, 14, 15, 17, 18, 24, 25, 27, 28, 34, 35, 37, 38]],
    },
    '19':{
        '19_1': [[7, 8, 14, 15, 17, 18, 20, 21, 27, 28, 30, 31, 33, 34, 40, 41, 43, 44, 46, 47]],
    }
    
}


def load_detection_model(opt, log_dir):
    if opt.model not in detection_models:
        raise ValueError('%s is not a valid model for the Object Detection task.'%opt.model)
    state_dict_path = './pretrained_parameters/%s_faster_rcnn_fpn_%s.pt'%(opt.model, opt.dataset)
    model = torch.load(state_dict_path)
    backbone = deepcopy(model.backbone)

    pruned_model_path = log_dir + 'model_pruned_finetuned_detection.pt'

    return model, backbone, pruned_model_path


def load_pose_model(opt, log_dir):
    if opt.model not in pose_models:
        raise ValueError('%s is not a valid model for the POSE task.'%opt.model)
    state_dict_path = './pretrained_parameters/%s_pose.pt'%(opt.model)
    model = torch.load(state_dict_path)
    backbone = deepcopy(model.base_net)
    heads = model.head_nets

    dataset_split_dir = './dataset_splits/%s_%s_it%i_bs%i/'%(opt.task, opt.dataset, opt.num_batches_ft, opt.bs_ft)
    if opt.orig_data_split:
        new_annotations_path = dataset_split_dir + 'annotations/person_keypoints_train2017.json'
        new_imgs_path = dataset_split_dir + 'images/train2017/'
    else:
        new_annotations_path = None
        new_imgs_path = None

    # Set hyperparameter
    pruned_model_path = log_dir + 'model_pruned_finetuned_pose.pt'
    pose_args = POSE_set_hyperparameter(pruned_model_path, new_imgs_path, new_annotations_path)
    pose_args.cif_base_stride = model.base_net.stride
    pose_args.cif_head_index = 0
    pose_args.caf_base_stride = model.base_net.stride
    pose_args.caf_head_index = 1

    return model, backbone, heads, pose_args, pruned_model_path, dataset_split_dir


def load_classification_models(opt):
    if opt.model not in classifiers.keys():
        raise ValueError('%s is not a valid model for the classification task.'%opt.model)
    state_dict_path = './pretrained_parameters/%s_%s.pt'%(opt.model, opt.dataset)
    model = classifiers[opt.model](num_classes=opt.num_classes)
    state_dict = torch.load(state_dict_path)
    model.load_state_dict(state_dict)
    backbone = deepcopy(model.backbone)
    heads = model.classifier

    return model, backbone, heads


if __name__ == '__main__':
    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='classification', choices=['classification', 'pose', 'detection'], help='Possible tasks: classification, detection, pose.')
    parser.add_argument('--prune', type=str2bool, default='True', help='Prune the model or use the saved pruned model.')
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50', 'vgg16', 'vgg19'], help='Define the model which should be used.')
    parser.add_argument('--device', type=str, default='cuda:0', help='Used device for the training.')
    parser.add_argument('--epochs_ft', type=int, default=200, help='Epochs to finetune the pruned model')
    parser.add_argument('--num_batches_ft', type=int, default=50, help='Number of iterations of one finetuning epoch.')
    parser.add_argument('--lr_ft', type=float, default=0.02, help='Learning rate for the finetuning.')
    parser.add_argument('--momentum_ft', type=float, default=0.9, help='Momentum for the finetuning.')
    parser.add_argument('--lr_steps_ft', type=int, default=[150], nargs='+', help='epochs at which to decay the learning rate')
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100', 'COCO', 'VOC'], help='Define the dataset on which the model was trained.')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes in the classification task.')
    parser.add_argument('--bs_ft', type=int, default=256, help='Batch size of the generated images in the finetuning.')
    parser.add_argument('--img_size', type=int, default=32, help='Image size of the generated images.')
    parser.add_argument('--log_dir_root', type=str, default='./runs', help='directory in which the results are saved.')
    parser.add_argument('--sparsity', type=float, default=0.65, help='Pruning sparsity for kernels with more the 60 filters.')
    parser.add_argument('--loss_func', type=str, default='intermediate', help='Possible loss functions are: mse, smoothl1, huber, l1, intermediate')
    parser.add_argument('--inter_loss_func', type=str, default='l1', help='Loss function for the intermediate loss. Possible: mse, huber, l1, smoothl1')
    parser.add_argument('--loss_weight_method', type=int, default=0, help='Loss weight method in the intermediate loss. Possible: 0,1,2')
    parser.add_argument('--use_DI', type=str2bool, default=True, help='Use the Deep Inversion method to generate images.')
    parser.add_argument('--pre_generated_images', type=str2bool, default=True, help='Use pre-generated images from zeroq.')
    parser.add_argument('--pre_generated_images_path', type=str, default='./generated_images', help='Path to the pregenerated images.')
    parser.add_argument('--pruning_stages', type=int, default=1, help='Number of pruning stages.')
    parser.add_argument('--layer_depending_sparsity', type=str2bool, default=True, help='Use the layer depending sparsity in the pruning')
    parser.add_argument('--pruning_method', type=str, default='L1', choices=['L1', 'BN'], help='For VGG both pruning methods are posible for ResNet only L1.')
    parser.add_argument('--intermediate_loss_mode', type=int, default=0, help='Mode of the intermediate loss. 0: every intermediate fm is used in the loss; 1: only every second is used.')
    parser.add_argument('--orig_data_split', type=str2bool, default=False, help='Use the same amount of original images as synthetic images')
    parser.add_argument('--create_new_orig_data_split', type=str2bool, default=False, help='Create a new dataset split from the original dataset.')
    opt = parser.parse_args()

    opt.sys_argv = 'python3 ' + ' '.join(sys.argv)

    if opt.task == 'pose' and 'resnet' in opt.model and opt.pruning_stages > 1:
        raise ValueError('ResNet pose pruning is only with 1 pruning stage possible.')

    if opt.task == 'detection' and opt.loss_func != 'intermediate':
        raise ValueError('In the Detection task only the intermediate loss is possible.')

    if opt.orig_data_split and opt.use_DI:
        raise ValueError('orig_data_split and use_DI can not set together.')

    log_dir = get_log_dir(opt.log_dir_root)
    tb_writer = SummaryWriter(log_dir=log_dir)
    res_file = log_dir + 'results.txt'
    #save hyperparameters
    with open(log_dir + 'opt.yaml', 'w') as f:
        yaml.dump(opt, f)

    # Set logger
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    mylogs = logging.getLogger(__name__)
    mylogs.setLevel(logging.INFO)
    file = logging.FileHandler(res_file)
    file.setLevel(logging.INFO)
    fileformat = logging.Formatter("%(message)s")
    file.setFormatter(fileformat)
    mylogs.addHandler(file)

    performance_pruned = 0
    dataset_split_dir = None


    mylogs.info('############################# LOAD MODELS #################################')
    # Load and prune the models
    if opt.task == 'pose':
        mylogs.info('Loading model in the pose estimation task.')
        model, backbone, heads, pose_args, pruned_model_path, dataset_split_dir = load_pose_model(opt, log_dir)

    elif opt.task == 'classification':
        mylogs.info('Loading model in the classification task.')
        model, backbone, heads = load_classification_models(opt)

    elif opt.task == 'detection':
        mylogs.info('Loading model in the detection task.')
        model, backbone, pruned_model_path = load_detection_model(opt, log_dir)
    
    else:
        raise ValueError('Unknown task: %s'%opt.task)

    mylogs.info('Number parameter whole model: %i'%count_parameters(model))
    mylogs.info('Number parameter unpruned backbone: %i'%count_parameters(backbone))
    mylogs.info('Number parameter heads: %i'%(count_parameters(model) - count_parameters(backbone)))
    mylogs.info('Ratio parameter in backbone and whole model: %f'%(count_parameters(backbone) / count_parameters(model)))

    if opt.use_DI:
        mylogs.info('############################## GENERATE IMAGES ##################################')
        regenerate = False
        method = 'DeepInversion'
        pre_generated_images_path = opt.pre_generated_images_path + '_%s_%s_%s_%s_%ipx_bs%i_it%i.pickle'%(method, opt.task, opt.model, opt.dataset, opt.img_size, opt.bs_ft, opt.num_batches_ft)

        if opt.pre_generated_images:
            if os.path.isfile(pre_generated_images_path):
                mylogs.info('Load the pre-generated images from %s.'%pre_generated_images_path)

                with open(pre_generated_images_path, 'rb') as f:
                    data = pickle.load(f)
                images = data['images']
                labels = data['labels']

                # Check if the hyp's are equal
                if opt.dataset != data['dataset'] or method != data['method'] or opt.bs_ft != data['batch_size'] or opt.num_batches_ft != data['num_batches_ft'] or opt.model != data['model'] or opt.img_size != data['image_size']:
                    regenerate = True
                    mylogs.info('Hyperparameter of the pre-generated images are not equal to the actual hyperparameters. Regenerate the images.')
            else:
                mylogs.info('File does not exist: %s. Generate the images new.'%pre_generated_images_path)
                regenerate = True

        # Generate new images
        if not opt.pre_generated_images or regenerate:
            mylogs.info('Generate new images with the %s approach.'%method)
            setting_id, jitter, parameters, coefficients = DeepInversionClass.get_hyperparameters(opt)

            # Citertion for the pseudo images generation
            criterion = nn.CrossEntropyLoss().to(opt.device)
            use_fp16 = True #False if opt.task == 'detection' else True
            in_model = deepcopy(backbone) if opt.task == 'detection' else deepcopy(model)

            DeepInversionEngine = DeepInversionClass(net_teacher=in_model.to(opt.device), path=log_dir, num_batches=opt.num_batches_ft,
                                        parameters=parameters, setting_id=setting_id, bs=opt.bs_ft, use_fp16=use_fp16,
                                        jitter=jitter, criterion=criterion, coefficients=coefficients, device=opt.device,
                                        task=opt.task, dataset=opt.dataset, num_classes=opt.num_classes)

            images, labels = DeepInversionEngine.generate_pseudo_dataset(output_path=pre_generated_images_path)


            data = {'images': images, 'labels': labels, 'dataset': opt.dataset, 'method': method, 'batch_size': opt.bs_ft, 'num_batches_ft': opt.num_batches_ft, 'model': opt.model, 'image_size': opt.img_size}
            with open(pre_generated_images_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        images = torch.cat(images, dim=0)
        labels = torch.cat(labels, dim=0)
        pseudo_dataset = PseudoImagesDataset(images, labels)
        pseudo_dataloader = torch.utils.data.DataLoader(pseudo_dataset, batch_size=opt.bs_ft, shuffle=True, num_workers=4)

    else:
        mylogs.info('############################## ORIGINAL IMAGES ##################################')
        # If no dataset split exists create one
        if dataset_split_dir is None:
            dataset_split_dir = './dataset_splits/%s_%s_it%i_bs%i/'%(opt.task, opt.dataset, opt.num_batches_ft, opt.bs_ft)

        if opt.orig_data_split and not opt.create_new_orig_data_split:
            if not os.path.isdir(dataset_split_dir):
                opt.create_new_orig_data_split = True
                mylogs.info('Create a new dataset split in: %s'%dataset_split_dir)
                
        if opt.orig_data_split and opt.create_new_orig_data_split:
            # create new dataset split
            if opt.task == 'pose':
                POSE_create_dataset_split(opt, dataset_split_dir)
                mylogs.info('Created a new dataset split for the pose task with %i images.'%(opt.num_batches_ft * opt.bs_ft))
            elif opt.task == 'detection':
                if opt.dataset == 'COCO':
                    detection_create_dataset_split_COCO(opt, dataset_split_dir)
                elif opt.dataset == 'VOC':
                    dataset_split_dir = detection_create_dataset_split_VOC(opt, dataset_split_dir)
            elif opt.task == 'classification':
                raise NotImplemented('Dataset splits for the classification are not implemented.')
        else:
            mylogs.info('Using the existing dataset split from: %s'%dataset_split_dir)

    # get layers to prune
    if opt.task == 'classification':
        if 'vgg' in opt.model:
            depth = int(opt.model.replace('vgg', ''))
            stages = '%i_%i'%(depth, opt.pruning_stages)
            layers2prune_all = pruning_stages_vgg[str(depth)][stages]
        elif 'resnet' in opt.model:
            stages = '%i_%i'%(int(opt.model.replace('resnet', '')), opt.pruning_stages)
            layers2prune_all = model.pruning_stages[stages]
    elif opt.task == 'pose':
        if opt.model == 'resnet50':
            layers2prune_all = [list(range(1000))] # prune everything in one stage
        else:
            raise ValueError('Only pruning ResNet50 is possible for pose.')
    elif opt.task == 'detection':
        if 'vgg' in opt.model:
            depth = int(opt.model.replace('vgg', ''))
            stages = '%i_%i'%(depth, opt.pruning_stages)
            layers2prune_all = pruning_stages_vgg[str(depth)][stages] # get pruning stages for the vgg
        else:
            layers2prune_all = [list(range(1000))]
    else:
        layers2prune_all = [list(range(1000))]

    mylogs.info('Prune for %i pruning stages'%opt.pruning_stages)
    for ps in range(opt.pruning_stages):
        layers2prune = layers2prune_all[ps]
        mylogs.info('################################# PRUNE #######################################')
        pruned_model, pruned_backbone = prune_backbone(opt, model, backbone, layers2prune, logger=mylogs.info)

        model = model.to(opt.device)
        pruned_model = pruned_model.to(opt.device)
        backbone = backbone.to(opt.device)
        pruned_backbone = pruned_backbone.to(opt.device)

        mylogs.info('############################ TEST ACCURACY AFTER PRUNING ##########################')
        if opt.task == 'pose':
            torch.save({'model': pruned_model, 'epoch': -1}, pruned_model_path)
            results = POSE_test(pruned_model_path, pose_args)
            performance_pruned = results['stats'][0]
            mylogs.info('mAP@0.5:0.95 of the pruned model: %f'%performance_pruned)

        elif opt.task == 'classification':
            accuracy_unpruned, class_accuracy_unpruned = CIFAR_test(model, opt)
            accuracy_pruned, class_accuracy_pruned = CIFAR_test(pruned_model, opt)
            mylogs.info('Accuracy of the original model: %f, \nAccuracy of the pruned model: %f'%(accuracy_unpruned, accuracy_pruned))

        elif opt.task == 'detection':
            torch.save(pruned_model, pruned_model_path)
            results = detection_test(opt, pruned_model_path)
            if opt.dataset == 'COCO':
                performance_pruned = results['bbox_mAP']
                mylogs.info('mAP@0.5:0.95 of the pruned model: %f'%performance_pruned)
            elif opt.dataset == 'VOC':
                performance_pruned = results['mAP']
                mylogs.info('mAP@0.5 of the pruned model: %f'%performance_pruned)
            


        mylogs.info('############################## LOSS #############################################')
        if 'vgg' in opt.model and opt.task == 'detection':
            layers2prune = [i + 2 for i in layers2prune]
        loss_func = get_loss(opt, backbone, pruned_backbone, logger=mylogs.info, layers2prune=layers2prune)


        mylogs.info('################################# FINETUNE ##############################################')
        # finetune the backbone with the generated image
        optimizer = torch.optim.SGD(pruned_backbone.parameters(), lr=opt.lr_ft, momentum=opt.momentum_ft, weight_decay=5e-4)

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, opt.lr_steps_ft, gamma=0.1)

        if opt.use_DI:
            mylogs.info('Use the inverted images for the finetuning.')
        else:
            mylogs.info('Use the original images for the finetuning.')

        for epoch in range(opt.epochs_ft):

            model = model.to(opt.device)
            pruned_model = pruned_model.to(opt.device)
            backbone = backbone.to(opt.device)
            pruned_backbone = pruned_backbone.to(opt.device)

            mean_loss = 0
            gt_labels = list()
            pruned_backbone.train()
            backbone.eval()

            if opt.use_DI:
                # Use the generated images for the finetuning

                for iter, (gen_imgs, gen_labels) in enumerate(pseudo_dataloader):

                    optimizer.zero_grad()
                    gen_imgs = gen_imgs.to(opt.device)

                    if isinstance(loss_func, IntermediateLoss):
                        loss_func.clear_hooks()

                    gt_preds = backbone(gen_imgs)
                    preds = pruned_backbone(gen_imgs)

                    if opt.task == 'classification':
                        gt_labels.append(gen_labels)

                    # Calculate loss
                    if isinstance(loss_func, IntermediateLoss):
                        loss = loss_func.get_intermediate_loss()
                    else:
                        loss = loss_func(preds, gt_preds.detach())

                    loss.backward()
                    optimizer.step()

                    mean_loss = (mean_loss * (iter) + loss.detach().cpu()) / (iter + 1)
            
            else: 
                # Use the original training images for the finetuning
                if opt.task == 'classification':
                    pruned_backbone, gt_labels, mean_loss = CIFAR_train(opt, optimizer, gt_labels, backbone, pruned_backbone, loss_func, mean_loss)
                elif opt.task == 'pose':
                    pruned_backbone, gt_labels, mean_loss = POSE_train(opt, epoch, optimizer, gt_labels, backbone, pruned_backbone, loss_func, pose_args, mean_loss)
                elif opt.task == 'detection':
                    pruned_backbone, gt_labels, mean_loss = detection_train(opt, epoch, optimizer, gt_labels, backbone, pruned_backbone, loss_func, mean_loss, dataset_split_dir=dataset_split_dir)

            # Save the model and print the results
            if opt.task == 'pose':
                pruned_model.base_net = pruned_backbone
                torch.save({'model': pruned_model, 'epoch': epoch}, pruned_model_path)
                if epoch % 25 == 0 or (epoch == opt.epochs_ft - 1):
                    results = POSE_test(pruned_model_path, pose_args)
                    performance_pruned = results['stats'][0]
                torch.save({'model': pruned_model.cpu().state_dict(), 'epoch': epoch}, pruned_model_path)
                
            elif opt.task == 'classification':
                gt_labels = torch.cat(gt_labels).cpu().detach().numpy()
                fig, ax = plt.subplots(1,1)
                ax.hist(gt_labels, bins=[0,1,2,3,4,5,6,7,8,9,10])
                pruned_model.backbone = pruned_backbone
                performance_pruned, acc_per_class = CIFAR_test(pruned_model, opt)
            
                tb_writer.add_figure('hist/class_acc', fig, epoch)
                for i, acc in enumerate(acc_per_class):
                    tb_writer.add_scalar('CA/accuracy_class%i'%i, acc, epoch)
                torch.save({'model': pruned_model.cpu().state_dict(), 'epoch': epoch}, log_dir + 'model_pruned_finetuned_classification.pt')

            elif opt.task == 'detection':
                pruned_model.backbone = pruned_backbone
                torch.save(pruned_model, pruned_model_path)
                if epoch % 10 == 0 or (epoch == opt.epochs_ft - 1):
                    results = detection_test(opt, pruned_model_path)
                    if opt.dataset == 'COCO':
                        performance_pruned = results['bbox_mAP']
                    elif opt.dataset == 'VOC':
                        performance_pruned = results['mAP']
                torch.save({'model': pruned_model.cpu().state_dict(), 'epoch': epoch}, pruned_model_path)

            tb_writer.add_scalar('finetuning_%i/overall_loss'%ps, mean_loss, epoch)
            tb_writer.add_scalar('finetuning_%i/performance'%ps, performance_pruned, epoch)
            tb_writer.add_scalar('lr_%i/lr_ft'%ps, optimizer.param_groups[0]['lr'], epoch)
            mylogs.info('Epoch %i; Loss: %f; Pruned performance: %f'%(epoch, mean_loss, performance_pruned))
            
            lr_scheduler.step()

        if ps + 1 < opt.pruning_stages:
            if isinstance(loss_func, IntermediateLoss):
                        loss_func.remove_hooks()
            model = deepcopy(pruned_model.cpu())
            backbone = deepcopy(pruned_backbone.cpu())

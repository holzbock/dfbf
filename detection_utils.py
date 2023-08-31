# Copyright (c) OpenMMLab. All rights reserved.

#######################################
######## File from mmDetection ########
###### changed/added by holzbock ######
#######################################

import argparse
import os
import os.path as osp
import time
import warnings
import sys
import pdb
from tqdm import tqdm
import shutil
import json
import random
from pycocotools.coco import COCO

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.utils import setup_multi_processes

from loss_functions import IntermediateLoss


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if 'pretrained' in cfg.model:
        cfg.model.pretrained = None
    elif 'init_cfg' in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None

    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    if args.checkpoint.endswith('.pth'):
        model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    else:
        model = torch.load(args.checkpoint)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    if args.checkpoint.endswith('.pth'):
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    else:
        checkpoint = torch.load(args.checkpoint).state_dict()
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  args.show_score_thr)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule', 'dynamic_intervals'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            metric = dataset.evaluate(outputs, **eval_kwargs)
            print(metric)
            metric_dict = dict(config=args.config, metric=metric)
            if args.work_dir is not None and rank == 0:
                mmcv.dump(metric_dict, json_file)

    return metric


def detection_test(opt, checkpoint_path):
    if opt.dataset == 'COCO':
        if opt.model == 'resnet50':
            config_file = 'pretrained_parameters/mmdet_config/faster_rcnn_r50_fpn_1x_coco.py'
        elif opt.model == 'vgg16':
            config_file = 'pretrained_parameters/mmdet_config/faster_rcnn_vgg16_bn_neck_1x_coco.py'
        elif opt.model == 'vgg19':
            config_file = 'pretrained_parameters/mmdet_config/faster_rcnn_vgg19_bn_neck_1x_coco.py'
        else:
            raise ValueError('Unknown model in the detection testing. Model: %s'%(opt.model))
        sys.argv = [sys.argv[0], config_file, checkpoint_path, '--eval', 'bbox']
    elif opt.dataset == 'VOC':
        if opt.model == 'resnet50':
            config_file = 'pretrained_parameters/mmdet_config/faster_rcnn_r50_fpn_1x_voc.py'
        elif opt.model == 'vgg16':
            config_file = 'pretrained_parameters/mmdet_config/faster_rcnn_vgg19_bn_neck_1x_voc.py'
        elif opt.model == 'vgg19':
            config_file = 'pretrained_parameters/mmdet_config/faster_rcnn_vgg19_bn_neck_1x_voc.py'
        else:
            raise ValueError('Unknown model in the detection testing. Model: %s'%(opt.model))
        sys.argv = [sys.argv[0], config_file, checkpoint_path, '--eval', 'mAP']
    else:
        raise ValueError('Unknown dataset in the detection testing. Dataset: %s, Model: %s'%(opt.dataset))
    
    stats = main()
    return stats

if __name__ == '__main__':
    main()

# python3 detection_utils.py configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth --eval bbox

def detection_train(opt, epoch, optimizer, gt_labels, backbone, pruned_backbone, loss_func, loss_mean, dataset_split_dir=None):

    # Get dataloader
    if opt.dataset == 'COCO' and opt.model == 'resnet50':
        cfg_path = 'pretrained_parameters/mmdet_config/faster_rcnn_r50_fpn_1x_coco.py'
    elif opt.dataset == 'VOC' and opt.model == 'resnet50':
        cfg_path = 'pretrained_parameters/mmdet_config/faster_rcnn_r50_fpn_1x_voc.py'
    elif opt.dataset == 'COCO' and opt.model == 'vgg16':
        cfg_path = 'pretrained_parameters/mmdet_config/faster_rcnn_vgg16_bn_neck_1x_coco.py'
    else:
        raise ValueError('Unknown model dataset combination in the detection training. Dataset: %s, Model: %s'%(opt.dataset, opt.model))
    cfg = Config.fromfile(cfg_path)
    # Set other training data path when training only with a part of the dataset
    if opt.orig_data_split:
        if dataset_split_dir is None:
            raise ValueError('For the split orig dataset training a dataset_split_dir must be defined.')
        if opt.dataset == 'COCO':
            cfg.data.train['ann_file'] = dataset_split_dir + 'annotations/instances_train2017.json'
            cfg.data.train['img_prefix'] = dataset_split_dir + 'train2017/'
        elif opt.dataset == 'VOC':
            cfg.data.train['dataset']['ann_file'] = dataset_split_dir
            cfg.data.train['times'] = 1

    dataset = build_dataset(cfg.data.train)
    samples_per_gpu = 2 #4 if torch.cuda.get_device_properties(opt.device).total_memory / 2**30 > 12. else 2
    trainloader = build_dataloader(dataset, samples_per_gpu=samples_per_gpu, workers_per_gpu=cfg.data.workers_per_gpu,
                                    dist=False, shuffle=True, runner_type='EpochBasedRunner')

    # Set the model to the right mode
    backbone.eval()
    pruned_backbone.train()

    # Train loop for each epoch
    for iter, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        optimizer.zero_grad()

        inputs = data['img'].data[0]
        inputs = inputs.to(opt.device)

        if isinstance(loss_func, IntermediateLoss):
            loss_func.clear_hooks()

        gt_preds = backbone(inputs)
        preds = pruned_backbone(inputs)

        # Calculate loss
        if isinstance(loss_func, IntermediateLoss):
            loss = loss_func.get_intermediate_loss()
        else:
            # NOTE this loss does not work because we get 4 different feature maps from the backbone
            loss = loss_func(preds, gt_preds.detach())

        loss.backward()
        optimizer.step()

        loss_mean = (loss_mean * (iter) + loss.detach().cpu()) / (iter + 1)

        # if iter > 20:
        #     break

    print('Mean epoch %i: %f'%(epoch, loss_mean))

    return pruned_backbone, gt_labels, loss_mean


def detection_create_dataset_split_COCO(opt, folder_name):
    annotations_path = './data-mscoco/annotations/instances_train2017.json'
    img_path = './data-mscoco/images/train2017/'
    # Remove old dir
    if os.path.isdir(folder_name):
        shutil.rmtree(folder_name)
    os.mkdir(folder_name)
    img_folder = folder_name + 'train2017/'
    os.mkdir(img_folder)
    annotations_folder = folder_name + 'annotations/'
    os.mkdir(annotations_folder)

    # Create new annotation file
    with open(annotations_path, 'r') as f:
        dataset = json.load(f)
    new_dataset = dict()
    new_dataset['info'] = dataset['info']
    new_dataset['licenses'] = dataset['licenses']
    new_dataset['categories'] = dataset['categories']

    # remove duplicate entries
    coco = COCO(annotations_path)

    def filter_image(image_id):
        ann_ids = coco.getAnnIds(imgIds=image_id)
        anns = coco.loadAnns(ann_ids)
        if not anns:
            return False
        else:
            return True
    
    ids_w_bb = coco.getImgIds()
    ids_w_bb = [image_id for image_id in ids_w_bb if filter_image(image_id)]

    number_imgs = opt.num_batches_ft * opt.bs_ft

    if number_imgs > len(ids_w_bb):
        raise ValueError('Not enough images with annotations in the COCO dataset. Needed images: %i, images: %i'%(number_imgs, len(ids_w_bb)))

    imgs_ids_split = random.sample(ids_w_bb, number_imgs)
    new_imgs_path = [os.path.join(img_folder, "%012d.jpg"%im) for im in imgs_ids_split]
    old_imgs_path = [os.path.join(img_path, "%012d.jpg"%im) for im in imgs_ids_split]

    for new, old in zip(new_imgs_path, old_imgs_path):
        shutil.copyfile(old, new)
    
    # Create new annotations and images
    images = list()
    for img in dataset['images']:
        if img['id'] in imgs_ids_split:
            images.append(img)
    new_dataset['images'] = images

    annotations = list()
    for ann in dataset['annotations']:
        if ann['image_id'] in imgs_ids_split:
            annotations.append(ann)
    new_dataset['annotations'] = annotations

    imgs_in_anns = list()
    for ann in annotations:
        if ann['image_id'] not in imgs_in_anns:
            imgs_in_anns.append(ann['image_id'])

    with open(folder_name + 'annotations/instances_train2017.json', 'w') as f:
        json.dump(new_dataset, f)

    return folder_name


def detection_create_dataset_split_VOC(opt, folder_name):
    split_name = '%s_%s_it%i_bs%i'%(opt.task, opt.dataset, opt.num_batches_ft, opt.bs_ft)
    annotations_path_07 = './data-voc/VOC2007/ImageSets/Main/trainval.txt'
    annotations_path_12 = './data-voc/VOC2012/ImageSets/Main/trainval.txt'
    img_path_07 = './data-voc/VOC2007/JPEGImages/'
    img_path_12 = './data-voc/VOC2012/JPEGImages/'
    # Remove old file
    split_file_07 = './data-voc/VOC2007/ImageSets/Main/trainval_%s.txt'%split_name
    split_file_12 = './data-voc/VOC2012/ImageSets/Main/trainval_%s.txt'%split_name
    if os.path.isfile(split_file_07):
        os.remove(split_file_07)
    if os.path.isfile(split_file_12):
        os.remove(split_file_12)

    # Get data
    with open(annotations_path_07, 'r') as f:
        data_07 = f.readlines()
    with open(annotations_path_12, 'r') as f:
        data_12 = f.readlines()
    data = data_07 + data_12

    number_imgs = opt.num_batches_ft * opt.bs_ft

    if number_imgs > len(data):
        raise ValueError('Not enough images in the VOC dataset. Needed images: %i, images: %i'%(number_imgs, len(data)))

    data_split = random.sample(data, number_imgs)

    split_07 = list()
    split_12 = list()
    for d in data_split:
        if d in data_07:
            split_07.append(d)
        if d in data_12:
            split_12.append(d)

    with open(split_file_07, 'w') as f:
        f.writelines(split_07)

    with open(split_file_12, 'w') as f:
        f.writelines(split_12)

    return [split_file_07, split_file_12]
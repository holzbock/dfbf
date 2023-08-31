import argparse
from collections import defaultdict
import glob
import json
import logging
import os
import sys
import time
import random
import shutil

import PIL
import thop
from tqdm import tqdm
import torch
from pycocotools.coco import COCO 

from openpifpaf import datasets, decoder, logger, network, show, visualizer, __version__
from openpifpaf.predictor import Predictor
import pdb

from torch.nn.modules import loss
from loss_functions import IntermediateLoss

LOG = logging.getLogger(__name__)


def default_output_name(args):
    output = '{}.eval-{}'.format(network.Factory.checkpoint, args.dataset)

    # coco
    if args.coco_eval_orientation_invariant or args.coco_eval_extended_scale:
        output += '-coco'
        if args.coco_eval_orientation_invariant:
            output += 'o'
        if args.coco_eval_extended_scale:
            output += 's'
    if args.coco_eval_long_edge is not None and args.coco_eval_long_edge != 641:
        output += '-cocoedge{}'.format(args.coco_eval_long_edge)

    # dense
    if args.dense_connections:
        output += '-dense'
        if args.dense_connections != 1.0:
            output += '{}'.format(args.dense_connections)

    return output


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass


def cli():
    parser = argparse.ArgumentParser(
        prog='python3 -m openpifpaf.eval',
        usage='%(prog)s [options]',
        description=__doc__,
        formatter_class=CustomFormatter,
    )
    parser.add_argument('--version', action='version',
                        version='OpenPifPaf {version}'.format(version=__version__))

    datasets.cli(parser)
    decoder.cli(parser)
    logger.cli(parser)
    network.Factory.cli(parser)
    Predictor.cli(parser, skip_batch_size=True, skip_loader_workers=True)
    show.cli(parser)
    visualizer.cli(parser)

    parser.add_argument('--output', default=None,
                        help='output filename without file extension')
    parser.add_argument('--skip-existing', default=False, action='store_true',
                        help='skip if output eval file exists already')
    parser.add_argument('--no-skip-epoch0', dest='skip_epoch0',
                        default=True, action='store_false',
                        help='do not skip eval for epoch 0')
    parser.add_argument('--watch', default=False, const=60, nargs='?', type=int,
                        help=('Watch a directory for new checkpoint files. '
                              'Optionally specify the number of seconds between checks.')
                        )
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    parser.add_argument('--write-predictions', default=False, action='store_true',
                        help='write a json and a zip file of the predictions')
    parser.add_argument('--show-final-image', default=False, action='store_true',
                        help='show the final image')
    parser.add_argument('--show-final-ground-truth', default=False, action='store_true',
                        help='show the final image with ground truth annotations')
    parser.add_argument('--n-images', default=None, type=int)
    parser.add_argument('--loader-warmup', default=3.0, type=float)
    args = parser.parse_args()

    logger.configure(args, LOG)

    # add args.device
    args.device = torch.device('cpu')
    args.pin_memory = False
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True
    LOG.debug('neural network device: %s', args.device)
    
    datasets.configure(args)
    decoder.configure(args)
    network.Factory.configure(args)
    Predictor.configure(args)
    show.configure(args)
    visualizer.configure(args)

    return args


def count_ops(model, height=641, width=641):
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, 3, height, width, device=device)
    gmacs, params = thop.profile(model, inputs=(dummy_input, ))
    LOG.info('GMACs = {0:.2f}, million params = {1:.2f}'.format(gmacs / 1e9, params / 1e6))
    return gmacs, params


def evaluate(args):
    # generate a default output filename
    if args.output is None:
        args.output = default_output_name(args)

    datamodule = datasets.factory(args.dataset)
    predictor = Predictor(checkpoint=args.checkpoint, head_metas=datamodule.head_metas)

    data_loader = datamodule.eval_loader()
    prediction_loader = predictor.enumerated_dataloader(enumerate(data_loader))
    if args.loader_warmup:
        LOG.info('Data loader warmup (%.1fs) ...', args.loader_warmup)
        time.sleep(args.loader_warmup)
        LOG.info('Done.')

    metrics = datamodule.metrics()
    total_start = time.perf_counter()
    loop_start = time.perf_counter()

    for image_i, (pred, gt_anns, image_meta) in enumerate(prediction_loader):
        LOG.info('image %d / %d, last loop: %.3fs, images per second=%.1f',
                 image_i, len(data_loader), time.perf_counter() - loop_start,
                 image_i / max(1, (time.perf_counter() - total_start)))
        loop_start = time.perf_counter()

        for metric in metrics:
            metric.accumulate(pred, image_meta, ground_truth=gt_anns)

        if args.show_final_image:
            # show ground truth and predictions on original image
            annotation_painter = show.AnnotationPainter()
            with open(image_meta['local_file_path'], 'rb') as f:
                cpu_image = PIL.Image.open(f).convert('RGB')

            with show.image_canvas(cpu_image) as ax:
                if args.show_final_ground_truth:
                    annotation_painter.annotations(ax, gt_anns, color='grey')
                annotation_painter.annotations(ax, pred)

        if args.n_images is not None and image_i >= args.n_images - 1:
            break

        # if image_i == 10:
        #     break

    total_time = time.perf_counter() - total_start

    # model stats
    counted_ops = list(count_ops(predictor.model_cpu))
    local_checkpoint = network.local_checkpoint_path(network.Factory.checkpoint)
    file_size = os.path.getsize(local_checkpoint) if local_checkpoint else -1.0

    # write
    additional_data = {
        'args': sys.argv,
        'version': __version__,
        'dataset': args.dataset,
        'total_time': total_time,
        'checkpoint': network.Factory.checkpoint,
        'count_ops': counted_ops,
        'file_size': file_size,
        'n_images': predictor.total_images,
        'decoder_time': predictor.total_decoder_time,
        'nn_time': predictor.total_nn_time,
    }

    metric_stats = defaultdict(list)
    for metric in metrics:
        if args.write_predictions:
            metric.write_predictions(args.output, additional_data=additional_data)

        this_metric_stats = metric.stats()
        assert (len(this_metric_stats.get('text_labels', []))
                == len(this_metric_stats.get('stats', [])))

        for k, v in this_metric_stats.items():
            metric_stats[k] = metric_stats[k] + v

    stats = dict(**metric_stats, **additional_data)

    LOG.info('stats:\n%s', json.dumps(stats, indent=4))
    LOG.info(
        'time per image: decoder = %.0fms, nn = %.0fms, total = %.0fms',
        1000 * stats['decoder_time'] / stats['n_images'],
        1000 * stats['nn_time'] / stats['n_images'],
        1000 * stats['total_time'] / stats['n_images'],
    )

    return stats


def POSE_test(checkpoint, args):
    stats = evaluate(args)
    return stats


def POSE_set_hyperparameter(checkpoint, new_imgs_path, new_annotations_path):
    # Setting the hyperparameters multiple times causes some issues.
    sys.argv = [sys.argv[0], '--output', 'outputs/benchmark-211207-164351/resnet50', 
                '--loader-workers=8', '--dataset=cocokp', 
                '--coco-no-eval-annotation-filter', '--force-complete-pose', 
                '--seed-threshold=0.2', '--decoder=cifcaf:0', '--checkpoint', 'resnet50',
                '--cocokp-upsample=2', '--cocokp-orientation-invariant=0.1', '--cocokp-extended-scale']
    sys.argv[-4] = checkpoint
    if new_imgs_path is not None and new_annotations_path is not None:
        sys.argv = sys.argv + ['--cocokp-train-annotations', new_annotations_path, 
                               '--cocokp-train-image-dir', new_imgs_path]
    args = cli()

    return args


def watch(args):
    assert args.output is None
    pattern = args.checkpoint
    evaluated_pattern = '{}*eval-{}.stats.json'.format(pattern, args.dataset)

    while True:
        # find checkpoints that have not been evaluated
        all_checkpoints = glob.glob(pattern)
        evaluated = glob.glob(evaluated_pattern)
        if args.skip_epoch0:
            all_checkpoints = [c for c in all_checkpoints
                               if not c.endswith('.epoch000')]
        checkpoints = [c for c in all_checkpoints
                       if not any(e.startswith(c) for e in evaluated)]
        LOG.info('%d checkpoints, %d evaluated, %d todo: %s',
                 len(all_checkpoints), len(evaluated), len(checkpoints), checkpoints)

        # evaluate all checkpoints
        for checkpoint in checkpoints:
            # reset
            args.output = None
            network.Factory.checkpoint = checkpoint

            evaluate(args)

        # wait before looking for more work
        time.sleep(args.watch)


def main():
    args = cli()

    if args.watch:
        watch(args)
    else:
        evaluate(args)


# Run the evaluation in the terminal.
# python3 ./pose_utils.py --output outputs/benchmark-211207-164351/resnet50 --loader-workers=8 --dataset=cocokp --coco-no-eval-annotation-filter --force-complete-pose --seed-threshold=0.2 --decoder=cifcaf:0 --checkpoint model_pruned_finetuned.pt
if __name__ == '__main__':
    main()



def POSE_train(opt, epoch, optimizer, gt_labels, backbone, pruned_backbone, loss_func, args, loss_mean):

    # Get dataloader
    datamodule = datasets.factory(args.dataset)
    datamodule.head_metas[0].base_stride = args.cif_base_stride
    datamodule.head_metas[0].head_index = 0
    datamodule.head_metas[1].base_stride = args.caf_base_stride
    datamodule.head_metas[1].head_index = 1
    bs = 8 if torch.cuda.get_device_properties(opt.device).total_memory / 2**30 > 12. else 2
    datamodule.batch_size = bs
    trainloader = datamodule.train_loader()

    # Set the model to the right mode
    backbone.eval()
    pruned_backbone.train()

    # Train loop for each epoch
    for iter, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        optimizer.zero_grad()

        inputs, output_cif_caf, meta_data = data # NOTE output_cif_caf has nan. There is somewhere an error FIX it before using it!!!
        inputs = inputs.to(opt.device)

        if isinstance(loss_func, IntermediateLoss):
            loss_func.clear_hooks()

        gt_preds = backbone(inputs).detach()
        preds = pruned_backbone(inputs)

        # Calculate loss
        if isinstance(loss_func, IntermediateLoss):
            loss = loss_func.get_intermediate_loss()
        else:
            loss = loss_func(preds, gt_preds)

        loss.backward()
        optimizer.step()

        loss_mean = (loss_mean * (iter) + loss.detach().cpu()) / (iter + 1)

    print('Mean epoch %i: %f'%(epoch, loss_mean))

    return pruned_backbone, gt_labels, loss_mean


def POSE_create_dataset_split(opt, folder_name):
    annotations_path = './data-mscoco/annotations/person_keypoints_train2017.json'
    img_path = './data-mscoco/images/train2017/'
    # Remove old dir
    if os.path.isdir(folder_name):
        shutil.rmtree(folder_name)
    os.mkdir(folder_name)
    img_folder = folder_name + 'images/'
    os.mkdir(img_folder)
    img_folder = folder_name + 'images/train2017/'
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

    coco = COCO(annotations_path)
    category_ids = [1]
    min_kp_anns = 1

    def filter_image(image_id):
        ann_ids = coco.getAnnIds(imgIds=image_id, catIds=category_ids)
        anns = coco.loadAnns(ann_ids)
        anns = [ann for ann in anns if not ann.get('iscrowd')]
        if not anns:
            return False
        kp_anns = [ann for ann in anns
                    if 'keypoints' in ann and any(v > 0.0 for v in ann['keypoints'][2::3])]
        return len(kp_anns) >= min_kp_anns
    
    ids_w_kps = coco.getImgIds(catIds=category_ids)
    ids_w_kps = [image_id for image_id in ids_w_kps if filter_image(image_id)]

    number_imgs = opt.num_batches_ft * opt.bs_ft

    if number_imgs > len(ids_w_kps):
        raise ValueError('Not enough images with keypoints. Needed images: %i, images: %i'%(number_imgs, len(ids_w_kps)))

    imgs_ids_split = random.sample(ids_w_kps, number_imgs)
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

    with open(folder_name + 'annotations/person_keypoints_train2017.json', 'w') as f:
        json.dump(new_dataset, f)

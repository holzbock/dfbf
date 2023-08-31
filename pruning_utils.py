import numpy as np
import torch
import random
from torchvision.models.resnet import Bottleneck, BasicBlock
import pdb
from copy import deepcopy
from utils import count_parameters
import os
try:
    from mmdet.models.backbones.resnet import Bottleneck as Bottleneck_mmdet
    from mmdet.models.backbones.resnet import BasicBlock as BasicBlock_mmdet
    from mmdet.models.backbones.vgg import VGG
except:
    print('mmDetection is not installed. The pruning of the FasterRCNN is not possible.')


def layer_eval(layer):
    element_squared = [e.item()**2 for e in layer.view(-1)]
    return sum(element_squared)


def get_prune_indices(channel):
    indexes = []
    sum = 0
    for idx, filter in enumerate(channel):
        sum = layer_eval(filter)
        indexes.append(sum)
    return indexes


def get_layers_2_prune(conv, method, sparsity, bn_layer=None, logger=print):
    num_channels_conv = conv.weight.shape[0]
    # get weight of each filter
    if method == "L1":
        sums = get_prune_indices(conv.weight)
        weights = np.array(sums)
    elif method == 'BN':
        if bn_layer == None or not isinstance(bn_layer, torch.nn.BatchNorm2d):
            raise ValueError('For the Batch Normalization Pruning a bn_layer must be defined.')
        weights = bn_layer.weight.clone().cpu().detach().numpy()
    else:
        raise ValueError('Unknown pruning method: %s'%method)

    # Determine sparsity
    if num_channels_conv > 1:
        sparsity_module = sparsity
    else:
        sparsity_module = 0.0

    # Determine channels to prune
    sparsity_module = int(num_channels_conv * sparsity_module)
    indexes = np.argpartition(weights, sparsity_module)[:sparsity_module]
    indices_pruned = indexes.tolist()

    # Check that we prune not to much layers
    logger("Pruned %i from %i kernels" %(len(indices_pruned), num_channels_conv))
    if len(indices_pruned) + 10 >= num_channels_conv:
        logger('Pruned to much filters. Pruned filters: %i, Overall filters: %i'%(len(indices_pruned), len(num_channels_conv)))
        indices_pruned = random.sample(indices_pruned, num_channels_conv - 10)
    indices_stayed = list(set(range(num_channels_conv)) - set(indices_pruned))

    return indices_stayed, indices_pruned


def remove_kernels(conv1, conv2, bn, indices_stayed):
    num_channels_stayed = len(indices_stayed)
    # remove kernels conv1 weights
    new_weight = conv1.weight[indices_stayed, ...].clone()
    del conv1.weight
    conv1.weight = torch.nn.Parameter(new_weight)
    conv1.out_channels = num_channels_stayed

    # remove kernels conv1 bias
    if conv1.bias is not None:
        new_bias = conv1.bias[indices_stayed, ...].clone()
        del conv1.bias
        conv1.bias = torch.nn.Parameter(new_bias)

    # adapt BN module
    new_running_mean = bn.running_mean[indices_stayed, ...].clone()
    new_running_var = bn.running_var[indices_stayed, ...].clone()
    new_weight = bn.weight[indices_stayed, ...].clone()
    if bn.bias is not None:
        new_bias = bn.bias[indices_stayed, ...].clone()
        del bn.bias
        bn.bias = torch.nn.Parameter(new_bias)
    del bn.running_mean
    del bn.running_var
    del bn.weight
    bn.register_buffer('running_mean', new_running_mean)
    bn.register_buffer('running_var', new_running_var)
    bn.weight = torch.nn.Parameter(new_weight)
    bn.num_features = num_channels_stayed

    # remove kernels conv2 weights
    new_weight = conv2.weight[:,indices_stayed, ...].clone()
    del conv2.weight
    conv2.weight = torch.nn.Parameter(new_weight)
    conv2.in_channels = num_channels_stayed


def prune_resnet(model, layers2prune, method="L1", sparsity=0.65, layer_depending_sparsity=False, logger=print):
    logger("Pruning with the %s method"%method)
    channels_pruned = []
    num_overall_channels = 0
    if method != 'L1':
        raise NotImplementedError('%s pruning for the ResNet is not implemented yet.'%method)

    for num, module in enumerate(model.modules()):
        if (isinstance(module, Bottleneck) or isinstance(module, Bottleneck_mmdet)) and num in layers2prune:
            conv1 = module.conv1
            bn1 = module.bn1
            conv2 = module.conv2
            bn2 = module.bn2
            conv3 = module.conv3
            bn3 = module.bn3
            num_overall_channels += conv1.out_channels
            num_overall_channels += conv2.out_channels

            # Get kernels to prune of conv1
            if layer_depending_sparsity:
                sp = sparsity * 1.3 if conv1.out_channels >= 500 else sparsity if conv1.out_channels > 200 and conv1.out_channels < 500 else sparsity * 0.4 
            else:
                sp = sparsity
            indices_stayed, indices_pruned = get_layers_2_prune(conv1, method, sp, logger=logger)
            channels_pruned.append(indices_pruned)
            
            # Remove the pruned kernels from conv1
            remove_kernels(conv1, conv2, bn1, indices_stayed)

            # Get kernels to prune of conv2
            if layer_depending_sparsity:
                sp = sparsity * 1.3 if conv2.out_channels >= 500 else sparsity if conv2.out_channels > 200 and conv2.out_channels < 500 else sparsity * 0.4 
            else:
                sp = sparsity
            indices_stayed, indices_pruned = get_layers_2_prune(conv2, method, sp, logger=logger)
            channels_pruned.append(indices_pruned)
            
            # Remove the pruned kernels from conv2
            remove_kernels(conv2, conv3, bn2, indices_stayed)
        
        elif (isinstance(module, BasicBlock) or isinstance(module, BasicBlock_mmdet)) and num in layers2prune:
            conv1 = module.conv1
            bn1 = module.bn1
            conv2 = module.conv2
            bn2 = module.bn2
            num_overall_channels += conv1.out_channels

            # Get kernels to prune of conv1
            if layer_depending_sparsity:
                sp = sparsity * 1.3 if conv1.out_channels >= 500 else sparsity if conv1.out_channels > 200 and conv1.out_channels < 500 else sparsity * 0.4 
            else:
                sp = sparsity
            indices_stayed, indices_pruned = get_layers_2_prune(conv1, method, sp, logger=logger)
            channels_pruned.append(indices_pruned)
            
            # Remove the pruned kernels from conv1
            remove_kernels(conv1, conv2, bn1, indices_stayed)

    logger('Overall channels: %i; pruned channels: %i; Pruning Ratio: %f'%(num_overall_channels, len(sum(channels_pruned, [])), len(sum(channels_pruned, [])) / num_overall_channels))

    return model


def prune_vgg(model, layers2prune, method="L1", sparsity=0.65, layer_depending_sparsity=False, logger=print):
    logger("Pruning with the %s method"%method)
    channels_pruned = []
    num_overall_channels = 0
    
    out_channels = 64
    indices_stayed = list(range(out_channels))

    if isinstance(model, VGG):
        model_it = model.features
        skip_modules = [0,1]
    else:
        skip_modules = [0, 1, 3, 4] # skip the first conv and the bn
        model_it = model
        
    for num, module in enumerate(model_it):
        
        if num in skip_modules:
            continue

        if isinstance(module, torch.nn.Conv2d):
            # remove input channels (which depends on the conv before)
            num_overall_channels += module.out_channels
            new_weight = module.weight[:,indices_stayed, ...].clone()
            del module.weight
            module.weight = torch.nn.Parameter(new_weight)
            module.in_channels = out_channels

            if num < len(model_it) - 4 and num in layers2prune: 
                # determine which channels to be pruned
                if layer_depending_sparsity:
                    sp = sparsity * 1.3 if module.out_channels > 500 else sparsity if module.out_channels > 200 and module.out_channels < 300 else sparsity * 0.4 
                else:
                    sp = sparsity

                indices_stayed, indices_pruned = get_layers_2_prune(module, method, sp, bn_layer=model_it[num+1], logger=logger)
                channels_pruned.append(indices_pruned)

                # remove the output channels
                new_weight = module.weight[indices_stayed, ...].clone()
                del module.weight
                module.weight = torch.nn.Parameter(new_weight)
                out_channels = len(indices_stayed)
                module.out_channels = out_channels

                if module.bias is not None:
                    new_bias = module.bias[indices_stayed, ...].clone()
                    del module.bias
                    module.bias = torch.nn.Parameter(new_bias)
            else:
                out_channels = module.weight.shape[0]
                indices_stayed = list(range(out_channels))
            
        if isinstance(module, torch.nn.BatchNorm2d) and num < len(model_it) - 4:
            # Remove pruned channels from the BN
            module.num_features = out_channels
            new_running_mean = module.running_mean[indices_stayed, ...].clone()
            new_running_var = module.running_var[indices_stayed, ...].clone()
            new_weight = module.weight[indices_stayed, ...].clone()
            del module.running_mean
            del module.running_var
            del module.weight
            module.register_buffer('running_mean', new_running_mean)
            module.register_buffer('running_var', new_running_var)
            module.weight = torch.nn.Parameter(new_weight)
            if module.bias is not None:
                new_bias = module.bias[indices_stayed, ...].clone()
                del module.bias
                module.bias = torch.nn.Parameter(new_bias)

    logger('Overall channels: %i; pruned channels: %i; Pruning Ratio: %f'%(num_overall_channels, len(sum(channels_pruned, [])), len(sum(channels_pruned, [])) / num_overall_channels))

    return model


def prune_backbone(opt, model, backbone, layers2prune, logger=print):
    logger('Start to prune the %s backbone.'%opt.model)
    pruned_model_path = './models_pruned/pruned_backbone_%s_%s_%s_sparsity_%.2f.pt'%(opt.model, opt.task, opt.dataset, opt.sparsity)

    # Load pruned model
    if os.path.isfile(pruned_model_path) and not opt.prune and opt.pruning_stages < 2:
        pruned_data = torch.load(pruned_model_path)
        if pruned_data['model_name'] == opt.model and pruned_data['sparsity'] == opt.sparsity and pruned_data['task'] == opt.task:
            pruned_backbone = pruned_data['backbone']
            logger('Loaded the pruned backbone from: %s'%pruned_model_path)
        else:
            opt.prune = True

    # Prune model
    if not os.path.isfile(pruned_model_path) or opt.prune or opt.pruning_stages > 1:
        if 'vgg' in opt.model:
            pruned_backbone = prune_vgg(deepcopy(backbone), layers2prune, method=opt.pruning_method, sparsity=opt.sparsity, layer_depending_sparsity=opt.layer_depending_sparsity, logger=logger)
        elif 'resnet' in opt.model:
            pruned_backbone = prune_resnet(deepcopy(backbone), layers2prune, method=opt.pruning_method, sparsity=opt.sparsity, layer_depending_sparsity=opt.layer_depending_sparsity, logger=logger)
        torch.save({'backbone': pruned_backbone, 'model_name': opt.model, 'sparsity': opt.sparsity, 'task': opt.task}, pruned_model_path)
        logger('Save the pruned backbone to: %s'%pruned_model_path)

    logger('Number parameter pruned backbone %i and unpruned backbone %i.'%(count_parameters(pruned_backbone), count_parameters(backbone)))
    logger('%f percent of the backbone parameters are pruned.'%(1 - count_parameters(pruned_backbone) / count_parameters(backbone)))

    pruned_model = deepcopy(model)
    if opt.task == 'pose':
        pruned_model.base_net = deepcopy(pruned_backbone)
    else:
        pruned_model.backbone = deepcopy(pruned_backbone)

    return pruned_model, pruned_backbone
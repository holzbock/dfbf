import torch
from torch import nn
import pdb
from torchvision.models.resnet import Bottleneck, BasicBlock
try:
    from mmdet.models.backbones.resnet import Bottleneck as Bottleneck_mmdet
    from mmdet.models.backbones.resnet import BasicBlock as BasicBlock_mmdet
except:
    print('mmDetection is not installed. The pruning of the FasterRCNN is not possible.')


class output_hook(object):
    def __init__(self):
        super(output_hook, self).__init__()
        self.outputs = None

    def hook(self, module, input, output):
        self.outputs = output

    def clear(self):
        self.outputs = None



def BN_loss(A, B):
    return (A - B).norm()**2 / B.size(0)


def get_BN_loss(teacher_model, gen_data, device):
    hooks, hook_handles, bn_stats = [], [], []
    eps = 1e-6
    teacher_model = teacher_model.to(device)
    teacher_model = teacher_model.eval()

    # get number of BatchNorm layers in the model
    layers = sum([
        1 if isinstance(layer, nn.BatchNorm2d) else 0
        for layer in teacher_model.modules()
    ])

    for n, m in teacher_model.named_modules():
        if isinstance(m, nn.Conv2d) and len(hook_handles) < layers:
            # register hooks on the convolutional layers to get the intermediate output after convolution and before BatchNorm.
            hook = output_hook()
            hooks.append(hook)
            hook_handles.append(m.register_forward_hook(hook.hook))
        if isinstance(m, nn.BatchNorm2d):
            # get the statistics in the BatchNorm layers
            bn_stats.append(
                (m.running_mean.detach().clone().flatten().to(device),
                 torch.sqrt(m.running_var +
                            eps).detach().clone().flatten().to(device)))
    assert len(hooks) == len(bn_stats)


    input_mean = torch.zeros(1, 3).to(device)
    input_std = torch.ones(1, 3).to(device)


    for hook in hooks:
        hook.clear()
    output = teacher_model(gen_data)
    mean_loss = 0
    std_loss = 0

    # compute the loss according to the BatchNorm statistics and the statistics of intermediate output
    for cnt, (bn_stat, hook) in enumerate(zip(bn_stats, hooks)):
        tmp_output = hook.outputs 
        bn_mean, bn_std = bn_stat[0], bn_stat[1]
        tmp_mean = torch.mean(tmp_output.view(tmp_output.size(0),
                                                tmp_output.size(1), -1),
                                dim=2)
        tmp_std = torch.sqrt(
            torch.var(tmp_output.view(tmp_output.size(0),
                                        tmp_output.size(1), -1),
                        dim=2) + eps)
        mean_loss += BN_loss(bn_mean, tmp_mean)
        std_loss += BN_loss(bn_std, tmp_std)
    tmp_mean = torch.mean(gen_data.view(gen_data.size(0), 3,
                                                -1),
                            dim=2)
    tmp_std = torch.sqrt(
        torch.var(gen_data.view(gen_data.size(0), 3, -1),
                    dim=2) + eps)
    mean_loss += BN_loss(input_mean, tmp_mean) # mean = 0
    std_loss += BN_loss(input_std, tmp_std) # std = 1
    total_loss = mean_loss + std_loss
    for handle in hook_handles:
        handle.remove()
    return total_loss, output


class IntermediateLoss():
    def __init__(self, teacher_model, student_model, loss_func=torch.nn.MSELoss(), loss_weight_method=0, 
                logger=print, model_type='resnet', layers2prune=None, mode=0):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.teacher_hooks, self.teacher_hook_handles = list(), list()
        self.student_hooks, self.student_hook_handles = list(), list()
        self.loss_func = loss_func
        self.mode = mode
        # Loss weight method: 0 -> no loss weight; 1 -> 0. loss weight; 2 -> 1. loss weight
        self.loss_weight_method = loss_weight_method    
        self.logger = logger
        if 'resnet' in model_type:
            self.model_type = 'resnet'
            self.register_hooks_resnet()
        elif 'vgg' in model_type:
            self.model_type = 'vgg'
            if layers2prune is None:
                raise ValueError('No layers2prune in the IntermediateLoss. For VGG model this have to be defined.')
            self.layers2prune = layers2prune + [2, 3, 5, 6] # ignore the first unpruned layers in the model
            self.register_hooks_vgg()
        else:
            raise ValueError('Unknown model tpye %s in the intermediate loss.'%model_type)
    
        self.clear_hooks()
        assert len(self.teacher_hooks) == len(self.student_hooks), 'Teacher and student has not the same number of intermediate layers for the intermediate loss.'
        self.num_intermediate_losses = len(self.teacher_hooks) 
        assert self.num_intermediate_losses >= 1, 'There are no intermediate layers found for the loss calculation. num_intermediate_loss = %i'%self.num_intermediate_losses
        self.logger('There are %i layers for the intermediate loss calculation.'%self.num_intermediate_losses)

        # Set which output stages are used for the loss calculation
        if self.mode == 0:
            self.output_stages = list(range(self.num_intermediate_losses))
        elif self.mode == 1:
            self.output_stages = list(range(0, self.num_intermediate_losses, 2))
            if self.num_intermediate_losses - 1 not in self.output_stages:
                self.output_stages.append(self.num_intermediate_losses - 1)
        else:
            raise ValueError('Unknown mode %i in the intermediate loss.'%self.mode)

        # define loss weights
        if self.loss_weight_method == 0:
            self.loss_weights = [1. for _ in range(1, self.num_intermediate_losses + 1)]
        elif self.loss_weight_method == 1:
            self.loss_weights = [i / self.num_intermediate_losses for i in range(1, self.num_intermediate_losses + 1)]
        elif self.loss_weight_method == 2:
            self.loss_weights = [(i / self.num_intermediate_losses + 1) for i in range(1, self.num_intermediate_losses + 1)]
        elif self.loss_weight_method == 3:
            self.loss_weights = [(i / self.num_intermediate_losses + 2) for i in range(1, self.num_intermediate_losses + 1)]
        elif self.loss_weight_method == 4:
            self.loss_weights = [(i / self.num_intermediate_losses) * 2 + 1 for i in range(1, self.num_intermediate_losses + 1)]
        elif self.loss_weight_method == 5:
            self.loss_weights = [(i / self.num_intermediate_losses) * 3 + 1 for i in range(1, self.num_intermediate_losses + 1)]
        elif self.loss_weight_method == 6:
            self.loss_weights = [(i / self.num_intermediate_losses) * 4 + 1 for i in range(1, self.num_intermediate_losses + 1)]
        elif self.loss_weight_method == 7:
            self.loss_weights = [(i / self.num_intermediate_losses) * 5 + 1 for i in range(1, self.num_intermediate_losses + 1)]
        elif self.loss_weight_method == 8:
            self.loss_weights = [(i / self.num_intermediate_losses) * 6 + 1 for i in range(1, self.num_intermediate_losses + 1)]
        elif self.loss_weight_method == 9:
            self.loss_weights = [(i / self.num_intermediate_losses) * 7 + 1 for i in range(1, self.num_intermediate_losses + 1)]
        elif self.loss_weight_method == 10:
            self.loss_weights = [(i / self.num_intermediate_losses) * 8 + 1 for i in range(1, self.num_intermediate_losses + 1)]
        elif self.loss_weight_method == 11:
            self.loss_weights = [(i / self.num_intermediate_losses) * 6 for i in range(1, self.num_intermediate_losses + 1)]
        else:
            raise ValueError('Loss weight method %i in the Intermediate Loss is not defined.'%self.loss_weight_method)


    def register_hooks_vgg(self):
        for it, ((t_name, t_param), (s_name, s_param)) in enumerate(zip(self.teacher_model.named_modules(), self.student_model.named_modules())):
            if isinstance(t_param, nn.BatchNorm2d) and isinstance(s_param, nn.BatchNorm2d) and t_name == s_name and it - 1 not in self.layers2prune:
                hook_t = output_hook()
                self.teacher_hooks.append(hook_t)
                self.teacher_hook_handles.append(t_param.register_forward_hook(hook_t.hook))
                hook_s = output_hook()
                self.student_hooks.append(hook_s)
                self.student_hook_handles.append(s_param.register_forward_hook(hook_s.hook))


    def register_hooks_resnet(self):
        # Calculate the loss for each Block
        for (t_name, t_param), (s_name, s_param) in zip(self.teacher_model.named_modules(), self.student_model.named_modules()):
            # Difference between both is:   Bottleneck has 3 convs -> register hook at the third conv
            #                               BasicBlock has 2 convs -> register hook at the second conv
            if (isinstance(t_param, Bottleneck) and isinstance(s_param, Bottleneck)) or (isinstance(t_param, Bottleneck_mmdet) and isinstance(s_param, Bottleneck_mmdet)) and t_name == s_name:
                hook_t = output_hook()
                self.teacher_hooks.append(hook_t)
                self.teacher_hook_handles.append(t_param.bn3.register_forward_hook(hook_t.hook))
                hook_s = output_hook()
                self.student_hooks.append(hook_s)
                self.student_hook_handles.append(s_param.bn3.register_forward_hook(hook_s.hook))
            elif (isinstance(t_param, BasicBlock) and isinstance(s_param, BasicBlock)) or (isinstance(t_param, BasicBlock_mmdet) and isinstance(s_param, BasicBlock_mmdet)) and t_name == s_name:
                hook_t = output_hook()
                self.teacher_hooks.append(hook_t)
                self.teacher_hook_handles.append(t_param.bn2.register_forward_hook(hook_t.hook))
                hook_s = output_hook()
                self.student_hooks.append(hook_s)
                self.student_hook_handles.append(s_param.bn2.register_forward_hook(hook_s.hook))


    def clear_hooks(self):
        for hook in self.teacher_hooks:
            hook.clear()
        for hook in self.student_hooks:
            hook.clear()


    def remove_hooks(self):
        for handle in self.teacher_hook_handles:
            handle.remove()
        for handle in self.student_hook_handles:
            handle.remove()


    def get_intermediate_loss(self):
        # NOTE if you use this loss don't forget to clear the loss before the forward path
        loss = 0

        for it, (t_hook, s_hook, loss_weight) in enumerate(zip(self.teacher_hooks, self.student_hooks, self.loss_weights)):
            if it in self.output_stages:
                t_out = t_hook.outputs
                s_out = s_hook.outputs
                loss = loss + loss_weight * self.loss_func(s_out, t_out)

        return loss


# Copied from: https://github.com/NVlabs/DeepInversion/blob/6d64b65c573a8229844c746d77993b2c0431a3e5/deepinversion.py#L55
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


def get_loss(opt, backbone, pruned_backbone, logger=print, layers2prune=None):

    if opt.loss_func == 'mse':
        loss_func = nn.MSELoss()
        logger('Using the MSE loss.')
    elif opt.loss_func == 'smoothl1':
        loss_func = nn.SmoothL1Loss(beta=0.015)
        logger('Using the Smooth L1 loss.')
    elif opt.loss_func == 'huber':
        loss_func = nn.HuberLoss(delta=0.015)
        logger('Using the Huber loss.')
    elif opt.loss_func == 'l1':
        loss_func = nn.L1Loss()
        logger('Using the L1 loss.')
    elif opt.loss_func == 'intermediate':
        if opt.inter_loss_func == 'mse':
            crit = nn.MSELoss()
        elif opt.inter_loss_func == 'l1':
            crit = nn.L1Loss()
        elif opt.inter_loss_func == 'huber':
            crit = nn.HuberLoss(delta=0.06)
        elif opt.inter_loss_func == 'smoothl1':
            crit = nn.SmoothL1Loss(beta=0.06)
        else:
            raise ValueError('Unknown criterion for the intermediate loss: %s'%opt.inter_loss_func)
        loss_func = IntermediateLoss(backbone, pruned_backbone, loss_func=crit, loss_weight_method=opt.loss_weight_method, 
                                    logger=logger, model_type=opt.model, layers2prune=layers2prune, mode=opt.intermediate_loss_mode)
        logger('Using the Intermediate loss with the %s loss.'%opt.inter_loss_func)
    else:
        raise ValueError('%s as loss not defined.'%opt.loss_func)

    return loss_func
        
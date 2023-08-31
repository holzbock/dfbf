# File adatpted from: https://github.com/huyvnphan/PyTorch_CIFAR10/blob/master/cifar10_models/resnet.py

import torch
import torch.nn as nn
import os
from torchvision.models.resnet import Bottleneck, BasicBlock

__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
]

pruning_stages = {
    '18': {
        '18_1': [[6, 12, 19, 28, 35, 44, 51, 60]],
        '18_2': [[12, 28, 44, 60], [6, 19, 35, 51]]
    }, 
    '34': {
        '34_1': [[6, 12, 18, 25, 34, 40, 46, 53, 62, 68, 74, 80, 86, 93, 102, 108]],
        '34_2': [[12, 25, 40, 53, 68, 80, 93, 108], [6, 18, 34, 46, 62, 74, 86, 102]]
    },
    '50': {
        '50_1': [list(range(300))]
    }
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        small_init_conv=True
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        # CIFAR10: kernel_size 7 -> 3, stride 2 -> 1, padding 3->1
        if small_init_conv:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        # END

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class ResNet_Classification(nn.Module):
    def __init__(
        self,
        block,
        layers,
        pruning_stages,
        num_classes=10,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        small_init_conv=True
    ):
        super(ResNet_Classification, self).__init__()
        
        self.pruning_stages = pruning_stages
        
        self.backbone = ResNet(block, layers, zero_init_residual=zero_init_residual, groups=groups, 
                                width_per_group=width_per_group, replace_stride_with_dilation=replace_stride_with_dilation, 
                                norm_layer=norm_layer, small_init_conv=small_init_conv)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512 * block.expansion, num_classes)


    def forward(self, x):
        x = self.backbone(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)

        return x


def _resnet(num_classes, arch, block, layers, pretrained, pruning_stages, device, small_init_conv, **kwargs):
    model = ResNet_Classification(block, layers, pruning_stages, num_classes=num_classes, small_init_conv=small_init_conv, **kwargs)
    if pretrained:
        script_dir = os.path.dirname(__file__)
        state_dict = torch.load(
            script_dir + "/state_dicts/" + arch + ".pt", map_location=device
        )
        model.load_state_dict(state_dict)
    return model


def resnet18(num_classes, small_init_conv=True, pretrained=False, device="cpu", **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(num_classes, 
        "resnet18", BasicBlock, [2, 2, 2, 2], pretrained, pruning_stages['18'], device, small_init_conv, **kwargs
    )


def resnet34(num_classes, small_init_conv=True, pretrained=False, device="cpu", **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(num_classes, 
        "resnet34", BasicBlock, [3, 4, 6, 3], pretrained, pruning_stages['34'], device, small_init_conv, **kwargs
    )


def resnet50(num_classes, small_init_conv=False, pretrained=False, device="cpu", **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(num_classes, 
        "resnet50", Bottleneck, [3, 4, 6, 3], pretrained, pruning_stages['50'], device, small_init_conv, **kwargs
    )

# File adatpted from: https://github.com/huyvnphan/PyTorch_CIFAR10/blob/master/cifar10_models/vgg.py
import os
import torch
import torch.nn as nn

__all__ = [
    "VGG",
    "vgg11_bn",
    "vgg13_bn",
    "vgg16_bn",
    "vgg19_bn",
]

# Layers to prune in each stage; first is the conv and second the BN; Name: ModelSize_PruningStages
pruning_stages = {
    '11':{
        '11_1': [list(range(100))],
        '11_2': [list(range(100))]
    },
    '13':{
        '13_1': [list(range(100))],
        '13_2': [list(range(100))]
    },
    '16':{
        '16_1': [[7, 8, 10, 11, 14, 15, 17, 18, 20, 21, 24, 25, 27, 28, 30, 31, 34, 35, 37, 38, 40]],
        '16_2': [[10, 11, 17, 18, 24, 25, 30, 31, 37, 38], [7, 8, 14, 15, 20, 21, 27, 28, 34, 35]]
    },
    '19':{
        '19_1': [[7, 8, 10, 11, 14, 15, 17, 18, 20, 21, 23, 24, 27, 28, 30, 31, 33, 34, 36, 37, 40, 41, 43, 44, 46, 47, 49]],
        '19_2': [[10, 11, 17, 18, 23, 24, 30, 31, 36, 37, 43, 44, 47], [7, 8, 14, 15, 20, 21, 27, 28, 33, 34, 40, 41, 46, 47]]
    }
    
}


class VGG(nn.Module):
    def __init__(self, backbone, pruning_stages, num_classes=10, init_weights=True):
        super(VGG, self).__init__()
        self.pruning_stages = pruning_stages
        self.backbone = backbone

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            # nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [ 64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M",],
    "E": [ 64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M",],
}


def _vgg(num_classes, arch, cfg, batch_norm, pretrained, pruning_stages, device, **kwargs):
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), pruning_stages, num_classes=num_classes, **kwargs)
    if pretrained:
        script_dir = os.path.dirname(__file__)
        state_dict = torch.load(
            script_dir + "/state_dicts/" + arch + ".pt", map_location=device
        )
        model.load_state_dict(state_dict)
    return model


def vgg11_bn(num_classes, pretrained=False, device="cpu", **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg(num_classes, "vgg11_bn", "A", True, pretrained, pruning_stages['11'], device, **kwargs)


def vgg13_bn(num_classes, pretrained=False, device="cpu", **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg(num_classes, "vgg13_bn", "B", True, pretrained, pruning_stages['13'], device, **kwargs)


def vgg16_bn(num_classes, pretrained=False, device="cpu", **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg(num_classes, "vgg16_bn", "D", True, pretrained, pruning_stages['16'], device, **kwargs)


def vgg19_bn(num_classes, pretrained=False, device="cpu", **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg(num_classes, "vgg19_bn", "E", True, pretrained, pruning_stages['19'], device, **kwargs)

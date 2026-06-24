import torch
import torch.nn as nn
import torchvision.models as tvm
from torch.ao.quantization import QuantStub, DeQuantStub
from torch.ao.nn.quantized import FloatFunctional


class _BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1    = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1      = nn.BatchNorm2d(planes)
        self.relu     = nn.ReLU(inplace=True)
        self.conv2    = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2      = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        self.add_op   = FloatFunctional()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.add_op.add(out, self.shortcut(x))
        return self.relu(out)


class ResNet20(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 16
        self.quant     = QuantStub()
        self.dequant   = DeQuantStub()
        self.conv1     = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1       = nn.BatchNorm2d(16)
        self.relu      = nn.ReLU(inplace=True)
        self.layer1    = self._layer(16, 3, stride=1)
        self.layer2    = self._layer(32, 3, stride=2)
        self.layer3    = self._layer(64, 3, stride=2)
        self.pool      = nn.AdaptiveAvgPool2d(1)
        self.fc        = nn.Linear(64, num_classes)

    def _layer(self, planes, n, stride):
        layers = [_BasicBlock(self.in_planes, planes, stride)]
        self.in_planes = planes
        for _ in range(n - 1):
            layers.append(_BasicBlock(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.quant(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.fc(self.pool(x).flatten(1))
        return self.dequant(x)


class _QuantBottleneck(nn.Module):
    def __init__(self, bottleneck):
        super().__init__()
        self.conv1      = bottleneck.conv1
        self.bn1        = bottleneck.bn1
        self.conv2      = bottleneck.conv2
        self.bn2        = bottleneck.bn2
        self.conv3      = bottleneck.conv3
        self.bn3        = bottleneck.bn3
        self.relu       = bottleneck.relu
        self.downsample = bottleneck.downsample
        self.add_op     = FloatFunctional()

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.add_op.add(out, identity)
        return self.relu(out)


class QuantResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        base = tvm.resnet50(weights=None)
        base.conv1   = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()
        base.fc      = nn.Linear(base.fc.in_features, num_classes)

        for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
            layer = getattr(base, layer_name)
            wrapped = nn.Sequential(*[_QuantBottleneck(b) for b in layer])
            setattr(base, layer_name, wrapped)

        self.quant   = QuantStub()
        self.dequant = DeQuantStub()
        self.conv1   = base.conv1
        self.bn1     = base.bn1
        self.relu    = base.relu
        self.maxpool = base.maxpool
        self.layer1  = base.layer1
        self.layer2  = base.layer2
        self.layer3  = base.layer3
        self.layer4  = base.layer4
        self.avgpool = base.avgpool
        self.fc      = base.fc

    def forward(self, x):
        x = self.quant(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return self.dequant(x)


class QuantVGG19(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        base = tvm.vgg19_bn(weights=None)
        self.quant      = QuantStub()
        self.dequant    = DeQuantStub()
        self.features   = base.features
        self.avgpool    = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return self.dequant(x)


class _QuantInvertedResidual(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.conv           = block.conv
        self.use_res_connect = block.use_res_connect
        self.add_op         = FloatFunctional()

    def forward(self, x):
        if self.use_res_connect:
            return self.add_op.add(x, self.conv(x))
        return self.conv(x)


class QuantMobileNetV2(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        base = tvm.mobilenet_v2(weights=None)
        base.features[0][0] = nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False)

        wrapped_features = []
        for module in base.features:
            if type(module).__name__ == "InvertedResidual":
                wrapped_features.append(_QuantInvertedResidual(module))
            else:
                wrapped_features.append(module)

        self.quant      = QuantStub()
        self.dequant     = DeQuantStub()
        self.features    = nn.Sequential(*wrapped_features)
        self.classifier  = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(base.last_channel, num_classes),
        )

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.classifier(x)
        return self.dequant(x)


class _QuantSqueezeExcitation(nn.Module):
    def __init__(self, se):
        super().__init__()
        self.avgpool = se.avgpool
        self.fc1 = se.fc1
        self.fc2 = se.fc2
        self.activation = se.activation
        self.scale_activation = se.scale_activation
        self.mul_op = FloatFunctional()

    def forward(self, x):
        scale = self.avgpool(x)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        scale = self.scale_activation(scale)
        return self.mul_op.mul(scale, x)


class _QuantMBConv(nn.Module):
    def __init__(self, block):
        super().__init__()
        wrapped_block = []
        for sub in block.block:
            if type(sub).__name__ == "SqueezeExcitation":
                wrapped_block.append(_QuantSqueezeExcitation(sub))
            else:
                wrapped_block.append(sub)
        self.block             = nn.Sequential(*wrapped_block)
        self.use_res_connect   = getattr(block, "use_res_connect", False)
        self.stochastic_depth  = block.stochastic_depth
        self.add_op            = FloatFunctional()

    def forward(self, x):
        result = self.block(x)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            return self.add_op.add(result, x)
        return result


def _replace_silu_with_relu(module):
    for name, child in module.named_children():
        if isinstance(child, nn.SiLU):
            setattr(module, name, nn.ReLU(inplace=True))
        else:
            _replace_silu_with_relu(child)


class QuantEfficientNetB0(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        base = tvm.efficientnet_b0(weights=None)
        fc   = base.features[0][0]
        base.features[0][0] = nn.Conv2d(fc.in_channels, fc.out_channels, 3, stride=1, padding=1, bias=False)
        _replace_silu_with_relu(base)

        wrapped_features = []
        for module in base.features:
            if type(module).__name__ == "Sequential":
                wrapped_sub = []
                for sub in module:
                    if type(sub).__name__ == "MBConv":
                        wrapped_sub.append(_QuantMBConv(sub))
                    else:
                        wrapped_sub.append(sub)
                wrapped_features.append(nn.Sequential(*wrapped_sub))
            else:
                wrapped_features.append(module)

        self.quant      = QuantStub()
        self.dequant    = DeQuantStub()
        self.features   = nn.Sequential(*wrapped_features)
        self.avgpool    = base.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(0.2, inplace=True),
            nn.Linear(base.classifier[-1].in_features, num_classes),
        )

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return self.dequant(x)


class _QuantCNBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block            = block.block
        self.layer_scale      = block.layer_scale
        self.stochastic_depth = block.stochastic_depth
        self.add_op           = FloatFunctional()

    def forward(self, x):
        result = self.layer_scale * self.block(x)
        result = self.stochastic_depth(result)
        return self.add_op.add(result, x)


class QuantConvNeXtTiny(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        base   = tvm.convnext_tiny(weights=None)
        in_ch  = base.features[0][0].in_channels
        out_ch = base.features[0][0].out_channels
        base.features[0][0] = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
        base.features[0][1] = nn.LayerNorm([out_ch, 32, 32])

        wrapped_features = []
        for module in base.features:
            if type(module).__name__ == "Sequential":
                wrapped_sub = []
                for sub in module:
                    if type(sub).__name__ == "CNBlock":
                        wrapped_sub.append(_QuantCNBlock(sub))
                    else:
                        wrapped_sub.append(sub)
                wrapped_features.append(nn.Sequential(*wrapped_sub))
            else:
                wrapped_features.append(module)

        self.quant      = QuantStub()
        self.dequant    = DeQuantStub()
        self.features   = nn.Sequential(*wrapped_features)
        self.avgpool    = base.avgpool
        self.classifier = nn.Sequential(
            base.classifier[0],
            base.classifier[1],
            nn.Linear(base.classifier[-1].in_features, num_classes),
        )

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return self.dequant(x)


_BUILDERS = {
    "resnet20":        ResNet20,
    "resnet50":        QuantResNet50,
    "vgg19":           QuantVGG19,
    "mobilenetv2":     QuantMobileNetV2,
    "efficientnet_b0": QuantEfficientNetB0,
    "convnext_tiny":   QuantConvNeXtTiny,
}


def get_model(arch, num_classes=10):
    arch = arch.lower()
    if arch not in _BUILDERS:
        raise ValueError(f"Unknown architecture '{arch}'. Choose from: {list(_BUILDERS.keys())}")
    return _BUILDERS[arch](num_classes)


def list_architectures():
    return list(_BUILDERS.keys())

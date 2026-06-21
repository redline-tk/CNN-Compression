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


def _build_resnet50(num_classes):
    m         = tvm.resnet50(weights=None)
    m.conv1   = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc      = nn.Linear(m.fc.in_features, num_classes)
    m.quant   = QuantStub()
    m.dequant = DeQuantStub()

    for module in m.modules():
        if type(module).__name__ == "Bottleneck":
            module.add_op = FloatFunctional()

            def make_forward(mod):
                def new_forward(x):
                    identity = x
                    out = mod.conv1(x)
                    out = mod.bn1(out)
                    out = mod.relu(out)
                    out = mod.conv2(out)
                    out = mod.bn2(out)
                    out = mod.relu(out)
                    out = mod.conv3(out)
                    out = mod.bn3(out)
                    if mod.downsample is not None:
                        identity = mod.downsample(x)
                    out = mod.add_op.add(out, identity)
                    out = mod.relu(out)
                    return out
                return new_forward

            module.forward = make_forward(module)

    orig_forward_impl = m._forward_impl

    def new_forward(x):
        x = m.quant(x)
        x = orig_forward_impl(x)
        return m.dequant(x)

    m.forward = new_forward
    return m


def _build_vgg19(num_classes):
    m            = tvm.vgg19_bn(weights=None)
    m.avgpool    = nn.AdaptiveAvgPool2d((1, 1))
    m.classifier = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(True),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes),
    )
    m.quant   = QuantStub()
    m.dequant = DeQuantStub()
    orig_features   = m.features
    orig_avgpool    = m.avgpool
    orig_classifier = m.classifier

    def new_forward(x):
        x = m.quant(x)
        x = orig_features(x)
        x = orig_avgpool(x)
        x = torch.flatten(x, 1)
        x = orig_classifier(x)
        return m.dequant(x)

    m.forward = new_forward
    return m


def _build_mobilenetv2(num_classes):
    m                 = tvm.mobilenet_v2(weights=None)
    m.features[0][0]  = nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False)
    m.classifier[-1]  = nn.Linear(m.last_channel, num_classes)
    m.quant   = QuantStub()
    m.dequant = DeQuantStub()

    for module in m.modules():
        if type(module).__name__ == "InvertedResidual" and getattr(module, "use_res_connect", False):
            module.add_op = FloatFunctional()

            def make_forward(mod):
                def new_forward(x):
                    return mod.add_op.add(x, mod.conv(x))
                return new_forward

            module.forward = make_forward(module)

    orig_features   = m.features
    orig_classifier = m.classifier

    def new_forward(x):
        x = m.quant(x)
        x = orig_features(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        x = orig_classifier(x)
        return m.dequant(x)

    m.forward = new_forward
    return m


def _build_efficientnet_b0(num_classes):
    m                = tvm.efficientnet_b0(weights=None)
    fc               = m.features[0][0]
    m.features[0][0] = nn.Conv2d(fc.in_channels, fc.out_channels, 3, stride=1, padding=1, bias=False)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    m.quant   = QuantStub()
    m.dequant = DeQuantStub()

    for module in m.modules():
        if type(module).__name__ == "MBConv" and getattr(module, "use_res_connect", False):
            module.add_op = FloatFunctional()

            def make_forward(mod):
                def new_forward(x):
                    result = mod.block(x)
                    result = mod.stochastic_depth(result)
                    return mod.add_op.add(result, x)
                return new_forward

            module.forward = make_forward(module)

    orig_features   = m.features
    orig_avgpool    = m.avgpool
    orig_classifier = m.classifier

    def new_forward(x):
        x = m.quant(x)
        x = orig_features(x)
        x = orig_avgpool(x)
        x = torch.flatten(x, 1)
        x = orig_classifier(x)
        return m.dequant(x)

    m.forward = new_forward
    return m


def _build_convnext_tiny(num_classes):
    m                = tvm.convnext_tiny(weights=None)
    in_ch            = m.features[0][0].in_channels
    out_ch           = m.features[0][0].out_channels
    m.features[0][0] = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
    m.features[0][1] = nn.LayerNorm([out_ch, 32, 32])
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    m.quant   = QuantStub()
    m.dequant = DeQuantStub()

    for module in m.modules():
        if type(module).__name__ == "CNBlock":
            module.add_op = FloatFunctional()

            def make_forward(mod):
                def new_forward(x):
                    result = mod.layer_scale * mod.block(x)
                    result = mod.stochastic_depth(result)
                    return mod.add_op.add(result, x)
                return new_forward

            module.forward = make_forward(module)

    orig_features   = m.features
    orig_avgpool    = m.avgpool
    orig_classifier = m.classifier

    def new_forward(x):
        x = m.quant(x)
        x = orig_features(x)
        x = orig_avgpool(x)
        x = orig_classifier(x)
        return m.dequant(x)

    m.forward = new_forward
    return m


_BUILDERS = {
    "resnet20":        ResNet20,
    "resnet50":        _build_resnet50,
    "vgg19":           _build_vgg19,
    "mobilenetv2":     _build_mobilenetv2,
    "efficientnet_b0": _build_efficientnet_b0,
    "convnext_tiny":   _build_convnext_tiny,
}


def get_model(arch, num_classes=10):
    arch = arch.lower()
    if arch not in _BUILDERS:
        raise ValueError(f"Unknown architecture '{arch}'. Choose from: {list(_BUILDERS.keys())}")
    return _BUILDERS[arch](num_classes)


def list_architectures():
    return list(_BUILDERS.keys())

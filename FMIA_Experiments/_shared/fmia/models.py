"""Architectures.  Every model takes images in [0, 1] and normalises internally, so
all perturbation budgets (ε) are in pixel units (k/255).

  resnet18, resnet34   CIFAR-style ResNet (3x3 stem, no max-pool at 32 px)
  wrn16_4, wrn28_2     WideResNet (pre-activation)
  vgg11, vgg16         VGG with batch-norm
  mobilenetv2          CIFAR-style MobileNetV2
  simplecnn            small 4-layer CNN (cheap; smoke tests)
  densenet121          torchvision DenseNet-121 (optionally ImageNet-pretrained: densenet121_pt)

Inputs larger than 32 px get a stride-2 stem (64 px) plus a max-pool (≥128 px),
so the compute per image stays close to CIFAR's.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Normalize(nn.Module):
    def __init__(self, mean, std, repeat_to=None):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))
        self.repeat_to = repeat_to

    def forward(self, x):
        x = (x - self.mean) / self.std
        if self.repeat_to and x.shape[1] != self.repeat_to:
            x = x.expand(-1, self.repeat_to, -1, -1)
        return x


def _stem_stride(img_size):
    return 1 if img_size <= 32 else 2


def _stem_pool(img_size):
    return nn.MaxPool2d(2) if img_size >= 128 else nn.Identity()


# ─── ResNet ───────────────────────────────────────────────────────────────────

class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.sc = nn.Sequential()
        if stride != 1 or cin != cout:
            self.sc = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        o = F.relu(self.b1(self.c1(x)))
        o = self.b2(self.c2(o))
        return F.relu(o + self.sc(x))


class ResNet(nn.Module):
    def __init__(self, blocks, num_classes, in_ch, img_size):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_ch, 64, 3, _stem_stride(img_size), 1, bias=False),
                                  nn.BatchNorm2d(64), nn.ReLU(inplace=True), _stem_pool(img_size))
        layers, cin = [], 64
        for i, (n, w) in enumerate(zip(blocks, (64, 128, 256, 512))):
            for k in range(n):
                layers.append(BasicBlock(cin, w, 2 if (k == 0 and i > 0) else 1))
                cin = w
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.layers(self.stem(x))
        return self.fc(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))


# ─── WideResNet ───────────────────────────────────────────────────────────────

class WideBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.b1 = nn.BatchNorm2d(cin)
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.sc = None
        if stride != 1 or cin != cout:
            self.sc = nn.Conv2d(cin, cout, 1, stride, bias=False)

    def forward(self, x):
        o = F.relu(self.b1(x))
        sc = self.sc(o) if self.sc is not None else x
        o = self.c1(o)
        o = self.c2(F.relu(self.b2(o)))
        return o + sc


class WideResNet(nn.Module):
    def __init__(self, depth, widen, num_classes, in_ch, img_size):
        super().__init__()
        n = (depth - 4) // 6
        widths = [16, 16 * widen, 32 * widen, 64 * widen]
        self.stem = nn.Sequential(nn.Conv2d(in_ch, widths[0], 3, _stem_stride(img_size), 1, bias=False),
                                  _stem_pool(img_size))
        layers, cin = [], widths[0]
        for i, w in enumerate(widths[1:]):
            for k in range(n):
                layers.append(WideBlock(cin, w, (1 if i == 0 else 2) if k == 0 else 1))
                cin = w
        self.layers = nn.Sequential(*layers)
        self.bn = nn.BatchNorm2d(cin)
        self.fc = nn.Linear(cin, num_classes)

    def forward(self, x):
        x = F.relu(self.bn(self.layers(self.stem(x))))
        return self.fc(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))


# ─── VGG ──────────────────────────────────────────────────────────────────────

VGG_CFG = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
}


class VGG(nn.Module):
    def __init__(self, name, num_classes, in_ch, img_size):
        super().__init__()
        layers, cin = [], in_ch
        if img_size > 32:
            layers += [nn.Conv2d(in_ch, 64, 3, _stem_stride(img_size), 1, bias=False),
                       nn.BatchNorm2d(64), nn.ReLU(inplace=True), _stem_pool(img_size)]
            cin = 64
        for v in VGG_CFG[name]:
            if v == "M":
                layers.append(nn.MaxPool2d(2))
            else:
                layers += [nn.Conv2d(cin, v, 3, 1, 1, bias=False), nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                cin = v
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.fc(torch.flatten(F.adaptive_avg_pool2d(self.features(x), 1), 1))


# ─── MobileNetV2 (CIFAR) ──────────────────────────────────────────────────────

class InvRes(nn.Module):
    def __init__(self, cin, cout, expansion, stride):
        super().__init__()
        mid = expansion * cin
        self.use_res = stride == 1 and cin == cout
        self.body = nn.Sequential(
            nn.Conv2d(cin, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU6(inplace=True),
            nn.Conv2d(mid, mid, 3, stride, 1, groups=mid, bias=False), nn.BatchNorm2d(mid), nn.ReLU6(inplace=True),
            nn.Conv2d(mid, cout, 1, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        o = self.body(x)
        return x + o if self.use_res else o


class MobileNetV2(nn.Module):
    CFG = [(1, 16, 1, 1), (6, 24, 2, 1), (6, 32, 3, 2), (6, 64, 4, 2), (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]

    def __init__(self, num_classes, in_ch, img_size):
        super().__init__()
        layers = [nn.Conv2d(in_ch, 32, 3, _stem_stride(img_size), 1, bias=False), nn.BatchNorm2d(32),
                  nn.ReLU6(inplace=True), _stem_pool(img_size)]
        cin = 32
        for t, c, n, s in self.CFG:
            for k in range(n):
                layers.append(InvRes(cin, c, t, s if k == 0 else 1))
                cin = c
        layers += [nn.Conv2d(cin, 1280, 1, bias=False), nn.BatchNorm2d(1280), nn.ReLU6(inplace=True)]
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        return self.fc(torch.flatten(F.adaptive_avg_pool2d(self.features(x), 1), 1))


# ─── Simple CNN ───────────────────────────────────────────────────────────────

class SimpleCNN(nn.Module):
    def __init__(self, num_classes, in_ch, img_size):
        super().__init__()
        s = _stem_stride(img_size)

        def blk(i, o, st=1):
            return [nn.Conv2d(i, o, 3, st, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]
        self.features = nn.Sequential(*blk(in_ch, 32, s), *blk(32, 64), nn.MaxPool2d(2),
                                      *blk(64, 128), nn.MaxPool2d(2), *blk(128, 256))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        return self.fc(torch.flatten(F.adaptive_avg_pool2d(self.features(x), 1), 1))


# ─── DenseNet-121 (torchvision) ───────────────────────────────────────────────

def _densenet121(num_classes, in_ch, img_size, pretrained):
    import torchvision
    w = torchvision.models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    m = torchvision.models.densenet121(weights=w)
    if img_size <= 64 or (in_ch != 3 and not pretrained):
        m.features.conv0 = nn.Conv2d(3 if pretrained else in_ch, 64, 3, 1 if img_size <= 32 else 2, 1, bias=False)
        m.features.pool0 = nn.Identity()
    m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    return m


# ─── factory ──────────────────────────────────────────────────────────────────

ARCHS = ["resnet18", "resnet34", "wrn16_4", "wrn28_2", "vgg11", "vgg16", "mobilenetv2",
         "simplecnn", "densenet121", "densenet121_pt"]


def build_model(arch, num_classes, in_ch, img_size, mean, std):
    pretrained = arch.endswith("_pt")
    if arch == "resnet18":
        net = ResNet([2, 2, 2, 2], num_classes, in_ch, img_size)
    elif arch == "resnet34":
        net = ResNet([3, 4, 6, 3], num_classes, in_ch, img_size)
    elif arch == "wrn16_4":
        net = WideResNet(16, 4, num_classes, in_ch, img_size)
    elif arch == "wrn28_2":
        net = WideResNet(28, 2, num_classes, in_ch, img_size)
    elif arch in VGG_CFG:
        net = VGG(arch, num_classes, in_ch, img_size)
    elif arch == "mobilenetv2":
        net = MobileNetV2(num_classes, in_ch, img_size)
    elif arch == "simplecnn":
        net = SimpleCNN(num_classes, in_ch, img_size)
    elif arch.startswith("densenet121"):
        net = _densenet121(num_classes, in_ch, img_size, pretrained)
    else:
        raise ValueError(f"unknown arch {arch!r}; choose from {ARCHS}")
    repeat = 3 if (pretrained and in_ch != 3) else None
    if repeat:
        mean, std = [sum(mean) / len(mean)] * 1, [sum(std) / len(std)] * 1
    return nn.Sequential(Normalize(mean, std, repeat_to=repeat), net)

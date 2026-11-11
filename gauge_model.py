"""
GAUGE-XAI model: a small 3D ResNet that outputs one malignancy logit for a 128^3, 1 mm patch.

Explainer-friendly by construction: no in-place ops, a separate ReLU module at every use, and the
last conv stage (layer4) at 8^3 (16 mm cells over the 128 mm FOV) as the Grad-CAM target.
The head concatenates global average and global max pooling: a small nodule is a tiny fraction of
the 128 mm field of view, and average pooling alone dilutes its signal.
Optional auxiliary outputs (n_aux, e.g. log nodule diameter) share the features during training only;
forward() always returns the malignancy logit alone, which is what the explainers attribute.
Optional centre prior (centre_prior=True): a fixed radial-distance map is appended inside the model as a
second channel. It is identical for every patch (no patient information); it only tells a translation-
equivariant, globally pooled network which object is the target (the one at the patch centre). The
explainers still take and attribute the CT channel alone.
"""
import torch
import torch.nn as nn


class BasicBlock3d(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(cout)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv3d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm3d(cout)
        self.relu2 = nn.ReLU()
        self.down = None
        if stride != 1 or cin != cout:
            self.down = nn.Sequential(nn.Conv3d(cin, cout, 1, stride, bias=False), nn.BatchNorm3d(cout))

    def forward(self, x):
        idt = x if self.down is None else self.down(x)
        y = self.relu1(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.relu2(y + idt)


class ResNet3d(nn.Module):
    """128^3 -> stem /2 (64^3) -> layer1 64^3 -> layer2 32^3 -> layer3 16^3 -> layer4 8^3 -> logit.
    With input_size=64 layer4 keeps stride 1, so layer4 is 8^3 either way."""

    def __init__(self, width=32, dropout=0.3, input_size=128, n_aux=0, centre_prior=False):
        super().__init__()
        w = width
        s4 = max(1, input_size // 64)
        self.centre_prior = centre_prior
        self._radial = {}
        self.stem = nn.Sequential(nn.Conv3d(2 if centre_prior else 1, w, 3, 2, 1, bias=False),
                                  nn.BatchNorm3d(w), nn.ReLU())
        self.layer1 = nn.Sequential(BasicBlock3d(w, w), BasicBlock3d(w, w))
        self.layer2 = nn.Sequential(BasicBlock3d(w, 2 * w, 2), BasicBlock3d(2 * w, 2 * w))
        self.layer3 = nn.Sequential(BasicBlock3d(2 * w, 4 * w, 2), BasicBlock3d(4 * w, 4 * w))
        self.layer4 = nn.Sequential(BasicBlock3d(4 * w, 8 * w, s4), BasicBlock3d(8 * w, 8 * w))
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.maxpool = nn.AdaptiveMaxPool3d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(16 * w, 1 + n_aux)

    def radial_map(self, x):
        """(1, 1, D, H, W) distance from the patch centre, 0 at the centre and 1 at a face centre."""
        key = (tuple(x.shape[2:]), x.device, x.dtype)
        if key not in self._radial:
            ax = [torch.linspace(-1, 1, n, device=x.device) for n in x.shape[2:]]
            zz, yy, xx = torch.meshgrid(*ax, indexing="ij")
            self._radial[key] = torch.sqrt(zz ** 2 + yy ** 2 + xx ** 2).to(x.dtype)[None, None]
        return self._radial[key]

    def forward_all(self, x):
        """x: (B, 1, D, H, W) CT patch -> (B, 1 + n_aux): column 0 is the malignancy logit."""
        if self.centre_prior:
            x = torch.cat([x, self.radial_map(x).expand(x.shape[0], -1, -1, -1, -1)], 1)
        x = self.layer4(self.layer3(self.layer2(self.layer1(self.stem(x)))))
        f = torch.cat([torch.flatten(self.avgpool(x), 1), torch.flatten(self.maxpool(x), 1)], 1)
        return self.fc(self.drop(f))

    def forward(self, x):
        return self.forward_all(x)[:, 0]


class DenseLayer3d(nn.Module):
    """BN-ReLU-1x1x1 conv (4g) - BN-ReLU-3x3x3 conv (g); output concatenated to the input."""

    def __init__(self, cin, growth):
        super().__init__()
        self.bn1, self.relu1 = nn.BatchNorm3d(cin), nn.ReLU()
        self.conv1 = nn.Conv3d(cin, 4 * growth, 1, bias=False)
        self.bn2, self.relu2 = nn.BatchNorm3d(4 * growth), nn.ReLU()
        self.conv2 = nn.Conv3d(4 * growth, growth, 3, 1, 1, bias=False)

    def forward(self, x):
        y = self.conv2(self.relu2(self.bn2(self.conv1(self.relu1(self.bn1(x))))))
        return torch.cat([x, y], 1)


def dense_block(cin, n_layers, growth):
    layers, c = [], cin
    for _ in range(n_layers):
        layers.append(DenseLayer3d(c, growth))
        c += growth
    return nn.Sequential(*layers), c


def transition(cin, cout):
    return nn.Sequential(nn.BatchNorm3d(cin), nn.ReLU(), nn.Conv3d(cin, cout, 1, bias=False), nn.AvgPool3d(2))


class DenseNet3d(nn.Module):
    """Second architecture family for the journal study (same input, head, centre prior and auxiliary output).
    128^3 -> stem /2 (64^3) -> transition (32^3) -> block 1 -> transition (16^3) -> layer3 (block 2)
    -> transition (8^3) -> layer4 (block 3, 8^3 = the Grad-CAM grid, as for the ResNet). With input_size=64
    the first transition is skipped, so layer4 is again 8^3."""

    def __init__(self, width=32, dropout=0.3, input_size=128, n_aux=0, centre_prior=False, growth=16,
                 block_layers=(4, 6, 8)):
        super().__init__()
        self.centre_prior = centre_prior
        self._radial = {}
        c = width
        self.stem = nn.Sequential(nn.Conv3d(2 if centre_prior else 1, c, 3, 2, 1, bias=False), nn.BatchNorm3d(c), nn.ReLU())
        pre = [transition(c, c)] if input_size >= 128 else []
        b1, c = dense_block(c, block_layers[0], growth)
        self.layer1 = nn.Sequential(*pre, b1)
        t2 = transition(c, c // 2)
        b2, c = dense_block(c // 2, block_layers[1], growth)
        self.layer3 = nn.Sequential(t2, b2)
        t3 = transition(c, c // 2)
        b3, c = dense_block(c // 2, block_layers[2], growth)
        self.layer4 = nn.Sequential(t3, b3, nn.BatchNorm3d(c), nn.ReLU())
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.maxpool = nn.AdaptiveMaxPool3d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * c, 1 + n_aux)

    radial_map = ResNet3d.radial_map

    def forward_all(self, x):
        if self.centre_prior:
            x = torch.cat([x, self.radial_map(x).expand(x.shape[0], -1, -1, -1, -1)], 1)
        x = self.layer4(self.layer3(self.layer1(self.stem(x))))
        f = torch.cat([torch.flatten(self.avgpool(x), 1), torch.flatten(self.maxpool(x), 1)], 1)
        return self.fc(self.drop(f))

    def forward(self, x):
        return self.forward_all(x)[:, 0]


ARCHS = {"resnet": ResNet3d, "densenet": DenseNet3d}


def build_model(width=32, dropout=0.3, input_size=128, n_aux=0, centre_prior=False, arch="resnet"):
    return ARCHS[arch](width=width, dropout=dropout, input_size=input_size, n_aux=n_aux, centre_prior=centre_prior)


def load_checkpoint(path, device="cpu"):
    """-> (model in eval mode, checkpoint dict). Used by the saliency scripts."""
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck.get("model_cfg", {})
    m = build_model(**cfg).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck

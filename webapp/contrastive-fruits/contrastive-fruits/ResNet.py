import torch
import torch.nn as nn
import torchvision.models as models


class ResNetBackbone(nn.Module):
    def __init__(self, pretrained=False, projection_size=128, projection_hidden=512):
        super().__init__()
        resnet = models.resnet18(pretrained=pretrained)
        # remove fc
        modules = list(resnet.children())[:-1]
        self.encoder = nn.Sequential(*modules)
        feat_dim = resnet.fc.in_features

        # projection head
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, projection_hidden),
            nn.ReLU(),
            nn.Linear(projection_hidden, projection_size)
        )

    def forward(self, x):
        # x: B x C x H x W
        h = self.encoder(x)  # B x feat_dim x 1 x 1
        h = h.view(h.size(0), -1)
        z = self.projector(h)
        z = nn.functional.normalize(z, p=2, dim=1)
        return z


def get_backbone(pretrained=False, projection_size=128, projection_hidden=512):
    return ResNetBackbone(pretrained=pretrained, projection_size=projection_size, projection_hidden=projection_hidden)

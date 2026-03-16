import torch
import torch.nn as nn
from torchvision import models


class CancerCellClassifier(nn.Module):

    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()

        if pretrained:
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            self.backbone = models.resnet18(weights=None)

        # 冻结backbone只训练fc，效果不如全部一起训
        # if freeze_backbone:
        #     for p in self.backbone.parameters():
        #         p.requires_grad = False

        # 替换fc层
        in_feat = self.backbone.fc.in_features  # 512 for resnet18
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_feat, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        # 加BatchNorm后效果没有明显提升
        # self.backbone.fc = nn.Sequential(
        #     nn.Linear(in_feat, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.4),
        #     nn.Linear(256, num_classes),
        # )

        for p in self.backbone.fc.parameters():
            p.requires_grad = True

    def forward(self, x):
        return self.backbone(x)

    # 两阶段训练用的，现在不冻结了用不到
    # def unfreeze_backbone(self, unfreeze_from="layer3"):
    #     layer_names = ["layer1", "layer2", "layer3", "layer4"]
    #     start = layer_names.index(unfreeze_from)
    #     to_unfreeze = layer_names[start:]
    #     for name, param in self.backbone.named_parameters():
    #         if any(ln in name for ln in to_unfreeze):
    #             param.requires_grad = True
    #     print(f"unfroze: {to_unfreeze}")


def get_model(num_classes=2, pretrained=True, device="cpu"):
    model = CancerCellClassifier(num_classes, pretrained)
    return model.to(device)

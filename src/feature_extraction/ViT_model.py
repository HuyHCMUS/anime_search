# ViT_model.py
import timm
import torch.nn as nn

class ViTModel(nn.Module):
    def __init__(self, num_classes=1000):
        super(ViTModel, self).__init__()
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

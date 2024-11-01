# Facenet.py
from facenet_pytorch import InceptionResnetV1
import torch.nn as nn

class FaceNetModel(nn.Module):
    def __init__(self, num_classes=4000):
        super(FaceNetModel, self).__init__()
        self.model = InceptionResnetV1(pretrained='vggface2').eval()
        self.model.classify = True
        self.model.logits = nn.Linear(self.model.logits.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

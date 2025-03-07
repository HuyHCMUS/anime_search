# Mô hình tạo embedding

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import sys
sys.path.append('/home/huy/project/anime_search_2/anime_search_engine')
from src.models.InceptionV1 import *


class CharacterEmbedder:
    def __init__(self, model_path='models/character_embedding.pt'):
        # Tải mô hình
        self.model = self._load_model(model_path)
        
        # Chuẩn bị transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _load_model(self,model_path):
        # Tải mô hình ResNet đã huấn luyện
        model = torch.load(
            model_path,
            map_location="cpu",
            weights_only=False  # Quan trọng: Cho phép tải toàn bộ mô hình
        )

        model.eval()  # Chế độ đánh giá
        
        return model
    
    def preprocess_image(self, image_file):
        # Tiền xử lý ảnh
        if isinstance(image_file, str):
            image = Image.open(image_file).convert('RGB')
        else:
            image = Image.fromarray(image_file).convert('RGB')
        return self.transform(image)
    
    def create_embedding(self, processed_image):
        # Tạo embedding từ ảnh
        with torch.no_grad():
            embedding = self.model(processed_image.unsqueeze(0))
        return embedding.numpy().flatten()

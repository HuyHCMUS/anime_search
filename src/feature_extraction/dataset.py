# dataset.py
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder


class AnimeDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.img_name = os.listdir(self.data_path)
        self.character, self.labels = self.get_character()

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, idx):
        #img_name = self.df.iloc[idx]['file_name']
        #character = self.df.iloc[idx]['character']
        img_path = os.path.join(self.data_path, self.img_name[idx])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
            
        return image, self.labels[idx]
    def get_character(self):
        character = []
        for img in self.img_name:
            character.append(img.split('-')[1][:-4])
            
        label_encoder = LabelEncoder()
        numeric_labels = label_encoder.fit_transform(character)
        return character, numeric_labels
        

# Define transforms
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
import sys
import torch
sys.path.append('/home/huy/project/anime_search_2/anime_search_engine')
from src.models.InceptionV1 import *
from src.models.embedding import CharacterEmbedder

from pathlib import Path


model_path = "/home/huy/project/anime_search_2/anime_search_engine/models/fine_tuned_facenetvx06080.pth"
    
character_embedder = CharacterEmbedder(model_path=model_path)

current_dir = Path(__file__).parent

# Use a test image from your data directory
image_path = current_dir /'image.jpg'  # Adjust path as needed

img_processed = character_embedder.preprocess_image(image_path)

embedding = character_embedder.create_embedding(img_processed)

print(embedding.shape)
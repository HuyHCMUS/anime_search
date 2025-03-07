# Kiểm thử xử lý ảnh
import sys
import io
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('/home/huy/project/anime_search_2/anime_search_engine')

from src.models.detector import AnimeCharacterDetector
from PIL import Image
from src.models.InceptionV1 import *
from src.models.embedding import CharacterEmbedder

# Get the current file's directory
current_dir = Path(__file__).parent

# Use a test image from your data directory
image_path = current_dir /'n'  # Adjust path as needed

detector = AnimeCharacterDetector('/home/huy/project/anime_search_2/anime_search_engine/models/animeheadsv3.pt')
model_path = "/home/huy/project/anime_search_2/anime_search_engine/models/fine_tuned_facenetvx06080.pth"
    
character_embedder = CharacterEmbedder(model_path=model_path)
img = Image.open(image_path)
img_array = np.array(img)
res = detector.detect_faces(img_array = img_array)

processed_image = character_embedder.preprocess_image(res[0]['face'])
print(processed_image.shape)
plt.imshow(res[0]['face'])
plt.show()


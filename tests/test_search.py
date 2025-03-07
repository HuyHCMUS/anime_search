# Kiểm thử tìm kiếm
import sys
sys.path.append('/home/huy/project/anime_search_2/anime_search_engine')
from src.search.similarity import SimilaritySearch
from src.models.embedding import CharacterEmbedder
from src.models.InceptionV1 import *

from pathlib import Path


model_path = "/home/huy/project/anime_search_2/anime_search_engine/models/fine_tuned_facenetvx06080.pth"
index_path = "/home/huy/project/anime_search_2/anime_search_engine/data/embeddings/faiss_index.index"
similarity_search = SimilaritySearch(index_path=index_path)
embedder = CharacterEmbedder(model_path)

current_dir = Path(__file__).parent
# Use a test image from your data directory
image_path = current_dir /'image.jpg'  # Adjust path as needed

img_processed = embedder.preprocess_image(image_path)
embedding = embedder.create_embedding(img_processed)
print(similarity_search.index.d)
similar_characters = similarity_search.find_similar(embedding)
print(similar_characters)
print(embedding.shape)


# Tìm kiếm tương tự
import faiss
import numpy as np
import json

class SimilaritySearch:
    def __init__(self, index_path='data/embeddings/faiss_index.index'):
        # Nạp chỉ mục Faiss
        self.index = faiss.read_index(index_path)
        
        # Nạp metadata nhân vật
        with open('data/embeddings/characters.json', 'r') as f:
            self.characters = json.load(f)
    
    def find_similar(self, query_embedding, top_k=5):
        # Tìm các nhân vật tương tự
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1), top_k
        )
        # Trả về thông tin các nhân vật
        similar_characters = [
            self.characters[idx] for idx in indices[0]
        ]
        
        return similar_characters
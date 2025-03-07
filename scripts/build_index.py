# Xây dựng FAISS index
import sys
import json
import torch
sys.path.append('/home/huy/project/anime_search_2/anime_search_engine')
from src.models.InceptionV1 import *
from src.models.embedding import CharacterEmbedder
import faiss

model_path = "/home/huy/project/anime_search_2/anime_search_engine/models/fine_tuned_facenetvx06080.pth"
character_embedder = CharacterEmbedder(model_path=model_path)

data_path = '/home/huy/project/anime_search_2/anime_search_engine/data/processed/face_crop'
files = os.listdir(data_path)

    

count = 0
metadata = []
faiss_index = faiss.IndexFlatL2(512)

for file_name in files:
    count +=1
    image_path = os.path.join(data_path,file_name)

    try:
        image_tensor = character_embedder.preprocess_image(image_path)
        with torch.no_grad():
            img_embedding = character_embedder.create_embedding(image_tensor)
   
    except:
        print(image_path)
    faiss_index.add(img_embedding.reshape(1, -1))
    metadata.append({"file_name": file_name})
    if count % 100 == 0:
        print(count)
faiss.write_index(faiss_index, "faiss_index3.index")
json.dump(metadata, open("metadata.json", "w"))

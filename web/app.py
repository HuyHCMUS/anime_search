# Ứng dụng chính

import streamlit as st
import sys
sys.path.append('/home/huy/project/anime_search_2/anime_search_engine')

from src.models.embedding import CharacterEmbedder
from src.models.detector import AnimeCharacterDetector
from src.search.similarity import SimilaritySearch
from src.models.InceptionV1 import *

from PIL import Image
import numpy as np
import os

def run_app():
    st.title("Anime Character Search")
    
    #Path
    embedding_model_path = "/home/huy/project/anime_search_2/anime_search_engine/models/fine_tuned_facenetvx06080.pth"
    detector_model_path = "/home/huy/project/anime_search_2/anime_search_engine/models/animeheadsv3.pt"
    index_path = "/home/huy/project/anime_search_2/anime_search_engine/data/embeddings/faiss_index3.index"
    data_path = '/home/huy/project/anime_search_2/anime_search_engine/data/processed/face_crop'

    # Khởi tạo các thành phần
    embedder = CharacterEmbedder(embedding_model_path)
    search = SimilaritySearch(index_path)
    detector = AnimeCharacterDetector(detector_model_path)

    similar_characters = []
    
    # Upload ảnh
    uploaded_file = st.file_uploader("Chọn ảnh nhân vật")
    
    if uploaded_file is not None:
        # Xử lý ảnh
        st.image(uploaded_file, caption='Ảnh đã tải lên.', width = 300)
        st.write("Đang tìm kiếm...")
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        faces = detector.detect_faces(img_array = img_array)

        for face in faces:
            img = face['face']
            processed_image = embedder.preprocess_image(img)
            embedding = embedder.create_embedding(processed_image)
            similar_characters+=list(search.find_similar(embedding))


        
        # Hiển thị kết quả
        
        if similar_characters:
            st.write("Các nhân vật tương tự:")
            
            # Tạo grid layout
            cols_per_row = 4
            
            # Xử lý từng nhóm 4 ảnh
            for i in range(0, len(similar_characters), cols_per_row):
                # Tạo 4 cột cho mỗi dòng
                cols = st.columns(cols_per_row)
                
                # Lấy nhóm 4 ảnh hiện tại
                current_batch = similar_characters[i:i + cols_per_row]
                
                # Hiển thị ảnh trong từng cột
                for j, character in enumerate(current_batch):
                    with cols[j]:
                        # Lấy tên file từ dictionary
                        file_name = character["file_name"]
                        
                        # Tạo đường dẫn đầy đủ đến file ảnh
                        image_path = os.path.join(data_path, file_name)
                        
                        # Hiển thị ảnh và tên nhân vật
                        character_name = file_name.split('-')[-1].replace('.jpg', '')
                        img = Image.open(image_path)
                        st.image(img, caption=character_name)

if __name__ == "__main__":
    run_app()

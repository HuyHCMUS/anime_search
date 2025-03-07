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

def run_app():
    st.title("Anime Character Search")
    
    #Path
    embedding_model_path = "/home/huy/project/anime_search_2/anime_search_engine/models/fine_tuned_facenetvx06080.pth"
    detector_model_path = "/home/huy/project/anime_search_2/anime_search_engine/models/animeheadsv3.pt"
    index_path = "/home/huy/project/anime_search_2/anime_search_engine/data/embeddings/faiss_index.index"


    # Khởi tạo các thành phần
    embedder = CharacterEmbedder(embedding_model_path)
    search = SimilaritySearch(index_path)
    detector = AnimeCharacterDetector(detector_model_path)

    similar_characters = []
    
    # Upload ảnh
    uploaded_file = st.file_uploader("Chọn ảnh nhân vật")
    
    if uploaded_file is not None:
        # Xử lý ảnh
        st.image(uploaded_file, caption='Ảnh đã tải lên.', use_column_width=True)
        st.write("Đang tìm kiếm...")
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        st.write(img_array.shape)
        faces = detector.detect_faces(img_array = img_array)

        for face in faces:
            img = face['face']
            processed_image = embedder.preprocess_image(img)
            embedding = embedder.create_embedding(processed_image)
            similar_characters+=list(search.find_similar(embedding))


        
        # Hiển thị kết quả
        st.write("Các nhân vật tương tự:",similar_characters)

if __name__ == "__main__":
    run_app()

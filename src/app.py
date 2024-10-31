import io
import torch
from torch import nn, optim
import timm
import faiss
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from PIL import Image
from fastapi.responses import JSONResponse
from face_detection.yolo_model import detect_face, load_model
from feature_extraction.models import AnimeClassifier
from feature_extraction.utils import extract_feature
import time
from facenet_pytorch import InceptionResnetV1
import cv2
from torchvision import transforms


def get_face_embedding(img,model):
    img = Image.fromarray(img)
    preprocess = transforms.Compose([
    transforms.Resize((224, 224)), # Resize về 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])    # Chuẩn hóa
    ])
    image_tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        embeddings = model(image_tensor) 

    return embeddings.detach().numpy()  # Convert to NumPy array # Convert to NumPy array

app = FastAPI()
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load các model phát hiện khuôn mặt và trích xuất đặc trưng
det_model = load_model('../models/anime_head_det.pt')  # Model phát hiện khuôn mặt
# CNN_model = AnimeClassifier(1000, pretrained=False)
# CNN_model.load_state_dict(torch.load('../models/anime_classifier.pth',map_location=torch.device('cpu'))) 
# CNN_model = InceptionResnetV1(pretrained='vggface2').eval()




# Khởi tạo mô hình để lấy embeddings
CNN_model = InceptionResnetV1(pretrained='vggface2')
CNN_model.classify = True  # Kích hoạt lớp phân loại
CNN_model.logits = nn.Linear(CNN_model.logits.in_features, 4000)

# Load các trọng số đã fine-tune
CNN_model.load_state_dict(torch.load('../models/fine_tuned_facenetv3.pth',map_location=torch.device('cpu')))
CNN_model.classify = False  # Kích hoạt lớp phân loại
CNN_model.eval()  # Chuyển mô hình sang chế độ đánh giá




def modifled_name(name):
    return name[:-5]

#CNN_model = CNN_model.to(device)
img_url = pd.read_csv('character_pic_url.csv')
img_url['char_url'] = img_url['char_url'].apply(modifled_name)
img_index = pd.read_csv('index5.csv')['img_index'].values
character_info = pd.read_csv('character_url.csv')
merged_info = pd.merge(img_url, character_info,  on="char_url", how="left")


@app.get("/")
async def home():
    return "Trang chủ"

@app.post("/search/")
async def search_similar_images(file: UploadFile = File(...)):
    start = time.time()
    # Đọc file ảnh từ phía client
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    # Chuyển đổi từ PIL Image sang numpy array cho model xử lý
    img_array = np.array(image)

    indices = []
    dist = []
    k = 30
    index_loaded = faiss.read_index("faiss_index5.index")

    # Phát hiện khuôn mặt và trích xuất đặc trưng
    face = detect_face(img_array, det_model)
    print(len(face))
    for f in face:
        # feature = extract_feature(face, CNN_model)
        feature = get_face_embedding(f,CNN_model)
        # Load Faiss index đã được huấn luyện và lưu
        

        # Tìm kiếm top-k ảnh tương tự
        
        distance, i = index_loaded.search(feature, k)
        indices.extend(i[0])
        dist.extend(distance[0])
    # Giả sử chúng ta có các đường dẫn ảnh ứng với index trong cơ sở dữ liệu
    similar_images = []
    id = []
    anime = []
    character = []
    dd = []
    for (idx,d) in zip(indices,dist):
        #image_url = int(img_url[int(idx)-2].split('-')[0][6:])
        index = img_index[int(idx)]
        dd.append(float(d))
        image_url = merged_info.iloc[index]['char_pic_url']
        character.append(merged_info.iloc[index]['Character'])
        anime.append(merged_info.iloc[index]['Anime'])
        similar_images.append(image_url)
        id.append(int(index))
    # Trả về danh sách các đường dẫn ảnh tương tự
    return JSONResponse(content={"similar_images": similar_images,
                                 "id": id,
                                 "character": character,
                                 "anime": anime,
                                 "distance": dd})
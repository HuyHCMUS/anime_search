# Mô hình nhận diện nhân vật
import cv2
import torch
import numpy as np
from ultralytics import YOLO

class AnimeCharacterDetector:
    def __init__(self, model_path='models/yolo_face_detector.pt'):
        """
        Khởi tạo detector khuôn mặt
        
        :param model_path: Đường dẫn tới model YOLO đã huấn luyện
        """
        # Nạp model YOLO
        self.model = YOLO(model_path)
        
        # Cài đặt ngưỡng tin cậy
        self.confidence_threshold = 0.7
        self.iou_threshold = 0.7
    
    def detect_faces(self, image_path = None, img_array = None):

        # Đọc ảnh
        img = img_array
        if isinstance(image_path, str):  # Check if image_path is a string path
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = img_array  # Use the provided numpy array directly
        
        # Phát hiện khuôn mặt
        results = self.model(img, conf=self.confidence_threshold, iou=self.iou_threshold)
        
        # Danh sách lưu khuôn mặt
        detected_faces = []
        
        # Xử lý các khuôn mặt được phát hiện
        for result in results:
            # Các box khuôn mặt
            boxes = result.boxes
            
            for box in boxes:
                # Tọa độ box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Cắt khuôn mặt
                face = img[y1:y2, x1:x2]
                
                # Resize về kích thước chuẩn
                face_resized = cv2.resize(face, (224, 224))
                
                detected_faces.append({
                    'face': face_resized,
                    'bbox': (x1, y1, x2, y2),
                    'confidence': float(box.conf)
                })
        
        return detected_faces
    
    # def detect_and_save_faces(self, image_path, output_dir='data/detected_faces/'):
    #     """
    #     Phát hiện và lưu các khuôn mặt
        
    #     :param image_path: Đường dẫn ảnh đầu vào
    #     :param output_dir: Thư mục lưu khuôn mặt
    #     """
    #     # Tạo thư mục nếu chưa tồn tại
    #     import os
    #     os.makedirs(output_dir, exist_ok=True)
        
    #     # Phát hiện khuôn mặt
    #     faces = self.detect_faces(image_path)
        
    #     # Lưu từng khuôn mặt
    #     saved_faces = []
    #     for i, face_data in enumerate(faces):
    #         # Tạo tên file
    #         filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_face_{i}.jpg"
    #         filepath = os.path.join(output_dir, filename)
            
    #         # Chuyển từ RGB sang BGR để lưu bằng OpenCV
    #         face_bgr = cv2.cvtColor(face_data['face'], cv2.COLOR_RGB2BGR)
    #         cv2.imwrite(filepath, face_bgr)
            
    #         # Lưu thông tin
    #         saved_faces.append({
    #             'filepath': filepath,
    #             'bbox': face_data['bbox'],
    #             'confidence': face_data['confidence']
    #         })
        
    #     return saved_faces
    

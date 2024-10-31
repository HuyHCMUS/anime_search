from ultralytics import YOLO
import os
import cv2
import matplotlib.pyplot as plt

def load_model(path='yolov8s.pt'):
    model = YOLO(path)
    return model

def train(model, config):
    model.train(data=config.yaml,
                epochs = config.epochs,
                batch = config.batch_size,
                imgsz = config.img_size,
                project=config.project_name,
                name = config.save_name)
    
def evaluate(model, config):
    model.val(
    project=config.project_name,
    name='val')

def predict_yolo(image_path, model):

    # Dự đoán đối tượng trong hình ảnh
    results = model(image_path)

    # Nếu muốn lấy bounding box và confidence scores
    boxes = results[0].boxes.xyxy  # Tọa độ bounding box (x1, y1, x2, y2)
    scores = results[0].boxes.conf  # Độ tin cậy cho mỗi box
    return boxes, scores

def visualize_prediction(image_path, boxes, scores):
  plt.figure(figsize=(10, 10))
  img = cv2.imread(image_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  for box, score in zip(boxes, scores):
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    cv2.putText(img, f"{score:.2f}", (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))

  plt.imshow(img)
  plt.show()

def detect_face(img, model):
    crop_imgs = []
    boxes, scores = predict_yolo(img, model)
    for box in boxes:
        x1, y1, x2, y2 = box
        cropped_img = img[int(y1):int(y2), int(x1):int(x2)]
        #cropped_img_bgr = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)
        crop_imgs.append(cropped_img)
    return crop_imgs

def main():
    config = {
        'yaml': '',
        'epochs': 10,
        'img_size': 640,
        'project_name': 'yolo',
        'save_name': 'face_det'
    }

    # #Train model: Fine-tune để phát hiện tốt khuôn mặt anime
    # model = load_model()
    # train(model,config)
    # evaluate(model, config)
    #==================
    # #Inference: Để cắt ra tất cả khuôn mặt trong các ảnh ban đầu
    # model_path ='' #Model đã fine-tune
    # path ='' # đường dẫn đến thư mục chứa ảnh gốc
    # output = '' # Đường dẫn đến thư mục ouput
    # model = load_model(model_path)
    # image_filenames = os.listdir(path)
    # for image_filename in image_filenames[:]:
    #     img_path = os.path.join(path, image_filename) 
    #     print(img_path)
    #     img = cv2.imread(img_path)
    #     try:
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     except:
    #         continue 
    #     boxes, scores = predict_yolo(img_path,model)
    #     for box, score in zip(boxes, scores):
    #         x1, y1, x2, y2 = box
    #         cropped_img = img[int(y1):int(y2), int(x1):int(x2)]
    #         output_path = os.path.join(output, f"{image_filename}")
    #         cropped_img_bgr = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)
    #         cv2.imwrite(output_path, cropped_img_bgr)

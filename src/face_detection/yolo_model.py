import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import argparse

class YOLOTrainer:
    def __init__(self, model_path='yolo11s.pt', yaml='', epochs=10, img_size=640, project_name='yolo', save_name='face_det'):
        self.model = YOLO(model_path)
        self.yaml = yaml
        self.epochs = epochs
        self.img_size = img_size
        self.project_name = project_name
        self.save_name = save_name

    def train(self):
        self.model.train(
            data=self.yaml,
            epochs=self.epochs,
            imgsz=self.img_size,
            project=self.project_name,
            name=self.save_name
        )
    
    def evaluate(self):
        self.model.val(project=self.project_name, name='val')

    def predict_yolo(self, image_path):
        results = self.model(image_path)
        boxes = results[0].boxes.xyxy  # Bounding box coordinates
        scores = results[0].boxes.conf  # Confidence scores
        return boxes, scores

    def visualize_prediction(self, image_path, boxes, scores):
        plt.figure(figsize=(10, 10))
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(img, f"{score:.2f}", (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
        
        plt.imshow(img)
        plt.show()

    def detect_face(self, image_path):
        img = cv2.imread(image_path)
        boxes, scores = self.predict_yolo(image_path)
        crop_imgs = [img[int(y1):int(y2), int(x1):int(x2)] for x1, y1, x2, y2 in boxes]
        return crop_imgs

def main(args):
    trainer = YOLOTrainer(
        model_path=args.model_path,
        yaml=args.yaml,
        epochs=args.epochs,
        img_size=args.img_size,
        project_name=args.project_name,
        save_name=args.save_name
    )

    if args.mode == 'train':
        trainer.train()
    elif args.mode == 'evaluate':
        trainer.evaluate()
    elif args.mode == 'predict':
        boxes, scores = trainer.predict_yolo(args.image_path)
        trainer.visualize_prediction(args.image_path, boxes, scores)

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='yolo11s.pt')
    parser.add_argument('--yaml', type=str, default='')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--img_size', type=int, default=640)
    parser.add_argument('--project_name', type=str, default='yolo')
    parser.add_argument('--save_name', type=str, default='face_det')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'predict', 'detect'], required=True)
    parser.add_argument('--image_path', type=str, help='Path to the image for prediction or detection')
    args = parser.parse_args()

    main(args)

import torch
from ultralytics import YOLO
import cv2
import os

ROOT = os.getcwd()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    model = YOLO("yolov8n.pt")  # Load pre-trained model
    model.train(data="/kaggle/input/parkingdata/datasets/data.yaml", epochs=50, imgsz=640, device=device)
    return model

def prediction(img_path, model):
    img = cv2.imread(img_path)
    results = model(img, device=device)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Get bbox coordinates
            cls = int(box.cls[0])  # Class ID
            conf = box.conf[0].item()  # Confidence score

            # Draw bounding box (avoid high CPU usage)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{cls}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save output image (instead of displaying directly)
    output_path = os.path.join(ROOT, "/result/detected.jpg")
    cv2.imwrite(output_path, img)

    print(f"Processed image saved at: {output_path}")
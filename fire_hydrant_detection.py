import torch
import cv2
import numpy as np
import os
from datetime import datetime
from detect_license_plate import detect_license_plate
from vehicle_classification import classify_vehicle, can_enter_public_office

# 현재 스크립트의 디렉토리 경로
current_dir = os.path.dirname(os.path.abspath(__file__))

# 경로 설정
img_path1 = os.path.join(current_dir, 'img/ok2.jpg')
img_path2 = os.path.join(current_dir, 'img/test2.jpg')
model_path1 = os.path.join(current_dir, 'car_epoch100.pt')
model_path2 = os.path.join(current_dir, 'fire_epoch200.pt')

# 모델 로드
model1 = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path1)
model2 = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path2)

# 이미지 로드
img1 = cv2.imread(img_path1)
img2 = cv2.imread(img_path2)
assert img1 is not None, 'Image 1 not found'
assert img2 is not None, 'Image 2 not found'

# 이미지 크기 조정 함수
def resize_image(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized

# 탐지 함수
def detect_and_draw(img, model, scale, label_suffix=""):
    results = model(img)
    threshold = 0.7
    detections = results.xyxy[0].cpu().numpy()
    filtered_detections = [det for det in detections if det[4] >= threshold]
    
    for det in filtered_detections:
        x1, y1, x2, y2, conf, cls = det
        label = f'{model.names[int(cls)]} {conf:.2f} {label_suffix}'
        color = (0, 255, 0)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(img, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    return img, filtered_detections

# 차량 번호판 탐지
img1_detected, detections1 = detect_and_draw(img1, model1, scale=100, label_suffix="Original")

# 소화전 탐지를 위해 이미지 크기 줄이기
img2_resized = resize_image(img2, scale_percent=50)
img2_detected, detections2 = detect_and_draw(img2_resized, model2, scale=50, label_suffix="Resized")

# 출력 디렉토리 생성
output_dir = os.path.join(current_dir, 'output')
os.makedirs(output_dir, exist_ok=True)

# 결과 이미지 저장
output_path1 = os.path.join(output_dir, 'test1_result.jpg')
output_path2 = os.path.join(output_dir, 'test2_result.jpg')

cv2.imwrite(output_path1, img1_detected)
cv2.imwrite(output_path2, img2_detected)

print(f"차량 번호판 탐지 결과 저장됨: {output_path1}")
print(f"소화전 탐지 결과 저장됨: {output_path2}")
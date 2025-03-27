import torch
import cv2
import numpy as np
import os
import datetime
import pandas as pd
from detect_license_plate import detect_license_plate
from vehicle_classification import classify_vehicle, can_enter_public_office

# 현재 스크립트의 디렉토리 경로
current_dir = os.path.dirname(os.path.abspath(__file__))

# 경로 설정
img_path = os.path.join(current_dir, 'img/able12.jpg')
model_path1 = os.path.join(current_dir, 'car_epoch100.pt')
model_path2 = os.path.join(current_dir, 'fire_epoch200.pt')

# 모델 로드
model1 = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path1)
model2 = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path2)

# 이미지 로드
img = cv2.imread(img_path)
assert img is not None, 'Image not found'

# 탐지
results1 = model1(img)
results2 = model2(img)

# 바운딩 박스를 그릴 결과 필터링 (임계값 0.7 이상)
threshold = 0.5
detections1 = results1.xyxy[0].cpu().numpy()
detections2 = results2.xyxy[0].cpu().numpy()

filtered_detections1 = [det for det in detections1 if det[4] >= threshold]
filtered_detections2 = [det for det in detections2 if det[4] >= threshold]

# 바운딩 박스 그리기 함수
def draw_boxes(detections, img, model):
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        label = f'{model.names[int(cls)]} {conf:.2f}'
        color = (0, 255, 0)  # Green color for bounding box
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(img, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

# 이미지에 바운딩 박스 그리기
draw_boxes(filtered_detections1, img, model1)
draw_boxes(filtered_detections2, img, model2)

# 결과 저장
output_path = os.path.join(current_dir, 'output/result.jpg')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
cv2.imwrite(output_path, img)
print(f"Result saved to {output_path}")

# 번호판 인식 및 바운딩 박스 그리기
license_plate_img, license_plate = detect_license_plate(img_path, model_path1)
print(f"Extracted license plate: {license_plate}")

# 결과 저장
output_path_license_plate = os.path.join(current_dir, 'output/result_license_plate.jpg')
cv2.imwrite(output_path_license_plate, license_plate_img)
print(f"Result saved to {output_path_license_plate}")

# 차량 분류 및 출입 가능 여부 확인
if license_plate:
    vehicle_type = classify_vehicle(license_plate)
    access_result = can_enter_public_office(license_plate)
    print(f"차량 유형: {vehicle_type}")
    print(f"출입 가능 여부: {access_result}")
else:
    vehicle_type = "알 수 없음"
    access_result = "알 수 없음"

# 소화전 탐지 결과 확인
fire_hydrant_detected = len(filtered_detections2) > 0

# 결과를 CSV 파일로 저장
results = [{
    "차량 번호": license_plate if license_plate else "인식 실패",
    "차량 유형": vehicle_type,
    "출입 가능 여부": access_result,
    "날짜": datetime.datetime.today().strftime('%Y-%m-%d'),  # 날짜 형식 지정
    "소화전 탐지": "탐지" if fire_hydrant_detected else "비탐지"
}]

csv_output_path = os.path.join(current_dir, 'output/result.csv')
df = pd.DataFrame(results)
df.to_csv(csv_output_path, index=False, encoding='utf-8-sig')
print(f"결과가 {csv_output_path}에 저장되었습니다.")

# infer.py 스크립트를 실행하여 TTS를 수행
import subprocess
subprocess.run(["python", os.path.join(current_dir, "vits/infer.py")])
wav_files = [f for f in os.listdir(os.path.join(current_dir, 'output')) if f.endswith('.wav')]
for wav_file in wav_files:
    wav_file_path = os.path.join(current_dir, 'output', wav_file)
    print(f"Playing {wav_file_path}")
    subprocess.run(['aplay', wav_file_path] if os.name != 'nt' else ['powershell', '-c', f'(New-Object Media.SoundPlayer "{wav_file_path}").PlaySync();'])
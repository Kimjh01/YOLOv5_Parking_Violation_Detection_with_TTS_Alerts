import torch
import cv2
import numpy as np

def detect_license_plate(img_path, model_path1):
    # 모델 로드
    model1 = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path1)

    # 이미지 로드
    img = cv2.imread(img_path)
    assert img is not None, 'Image not found'

    # 탐지
    results1 = model1(img)

    # 바운딩 박스를 그릴 결과 필터링 (임계값 0.7 이상)
    threshold = 0.7
    detections1 = results1.xyxy[0].cpu().numpy()
    filtered_detections1 = [det for det in detections1 if det[4] >= threshold]

    # 문자 좌표를 왼쪽에서 오른쪽으로 정렬
    sorted_detections = sorted(filtered_detections1, key=lambda x: x[0])

    # 라벨 매핑 (영어 -> 한글)
    label_map = {
        'beo': '버', 'bo': '보', 'bu': '부', 'da': '다', 'deo': '더', 'do': '도', 'du': '두',
        'eo': '어', 'ga': '가', 'geo': '거', 'go': '고', 'gu': '구', 'ha': '하', 'heo': '허',
        'ho': '호', 'jeo': '저', 'jo': '조', 'ju': '주', 'la': '라', 'leo': '러', 'lo': '로',
        'lu': '루', 'ma': '마', 'meo': '머', 'mo': '모', 'mu': '무', 'na': '나', 'neo': '너',
        'no': '노', 'nu': '누', 'o': '오', 'seo': '서', 'so': '소', 'su': '수', 'u': '우'
    }

    # 라벨 추출 및 변환
    labels = [label_map.get(model1.names[int(det[5])], model1.names[int(det[5])]) for det in sorted_detections]

    # 번호판 문자열 생성
    license_plate = ''.join(labels)

    # 불필요한 부분 제거
    if license_plate.startswith('license_plate'):
        license_plate = license_plate[len('license_plate'):]

    print(f"Extracted license plate: {license_plate}")

    # 바운딩 박스 그리기 함수
    def draw_boxes(detections, img):
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            label = f'{model1.names[int(cls)]} {conf:.2f}'
            color = (0, 255, 0)  # Green color for bounding box
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(img, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    draw_boxes(sorted_detections, img)
    
    return img, license_plate
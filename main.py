import torch
import cv2
import numpy as np
import os
import datetime
import pandas as pd
import argparse
import subprocess
from vehicle_classification import classify_vehicle, can_enter_public_office
from vits.infer import vits

def get_args():
    parser = argparse.ArgumentParser(description='YOLOv5 License Plate and Fire Hydrant Detection')
    parser.add_argument('--image', type=str, default='img/able12.jpg', help='path to input image')
    parser.add_argument('--car_model', type=str, default='car_epoch100.pt', help='path to car detection model')
    parser.add_argument('--fire_model', type=str, default='fire_epoch200.pt', help='path to fire hydrant detection model')
    parser.add_argument('--vits_checkpoint', type=str, default='vits/checkpoints/lasttry/G_51000.pth', help='path to vits checkpoint')
    parser.add_argument('--vits_config', type=str, default='vits/checkpoints/lasttry/config.json', help='path to vits config')
    parser.add_argument('--output_dir', type=str, default='output', help='path to output directory')
    parser.add_argument('--threshold', type=float, default=0.5, help='confidence threshold for detection')
    return parser.parse_args()

def detect_objects(img, model, threshold):
    results = model(img)
    detections = results.xyxy[0].cpu().numpy()
    filtered_detections = [det for det in detections if det[4] >= threshold]
    return filtered_detections

def draw_boxes(img, detections, model):
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        label = f'{model.names[int(cls)]} {conf:.2f}'
        color = (0, 255, 0)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return img

def extract_license_plate(detections, model):
    sorted_detections = sorted(detections, key=lambda x: x[0])
    label_map = {
        'beo': '버', 'bo': '보', 'bu': '부', 'da': '다', 'deo': '더', 'do': '도', 'du': '두',
        'eo': '어', 'ga': '가', 'geo': '거', 'go': '고', 'gu': '구', 'ha': '하', 'heo': '허',
        'ho': '호', 'jeo': '저', 'jo': '조', 'ju': '주', 'la': '라', 'leo': '러', 'lo': '로',
        'lu': '루', 'ma': '마', 'meo': '머', 'mo': '모', 'mu': '무', 'na': '나', 'neo': '너',
        'no': '노', 'nu': '누', 'o': '오', 'seo': '서', 'so': '소', 'su': '수', 'u': '우'
    }
    labels = [label_map.get(model.names[int(det[5])], model.names[int(det[5])]) for det in sorted_detections]
    license_plate = ''.join(labels)
    if license_plate.startswith('license_plate'):
        license_plate = license_plate[len('license_plate'):]
    return license_plate

def main(args):

    os.makedirs(args.output_dir, exist_ok=True)
    model1 = torch.hub.load('ultralytics/yolov5', 'custom', path=args.car_model)
    model2 = torch.hub.load('ultralytics/yolov5', 'custom', path=args.fire_model)
    tts_model = vits(args.vits_checkpoint, args.vits_config)
    img = cv2.imread(args.image)
    assert img is not None, f'Image not found at {args.image}'
    car_detections = detect_objects(img, model1, args.threshold)
    fire_detections = detect_objects(img, model2, args.threshold)
    img_with_boxes = draw_boxes(img.copy(), car_detections, model1)
    img_with_boxes = draw_boxes(img_with_boxes, fire_detections, model2)
    result_img_path = os.path.join(args.output_dir, 'result.jpg')
    cv2.imwrite(result_img_path, img_with_boxes)
    print(f"Result image saved to {result_img_path}")

    license_plate = extract_license_plate(car_detections, model1)
    print(f"Extracted license plate: {license_plate}")

    if license_plate:
        vehicle_type = classify_vehicle(license_plate)
        access_result = can_enter_public_office(license_plate)
    else:
        vehicle_type = "알 수 없음"
        access_result = "알 수 없음"


    fire_hydrant_detected = len(fire_detections) > 0
    results = [{
        "차량 번호": license_plate if license_plate else "인식 실패",
        "차량 유형": vehicle_type,
        "출입 가능 여부": access_result,
        "날짜": datetime.datetime.today().strftime('%Y-%m-%d'),
        "소화전 탐지": "탐지" if fire_hydrant_detected else "비탐지"
    }]
    csv_output_path = os.path.join(args.output_dir, 'result.csv')
    df = pd.DataFrame(results)
    df.to_csv(csv_output_path, index=False, encoding='utf-8-sig')
    print(f"Results saved to {csv_output_path}")
    messages = []
    def add_space_to_license_plate(plate):
        return '  '.join(list(plate))

    for index, row in df.iterrows():
        lp_spaced = add_space_to_license_plate(row["차량 번호"])
        fire_detected = row["소화전 탐지"] == "탐지"
        access = row["출입 가능 여부"]
        if fire_detected:
            messages.append(f"{lp_spaced} 번님 옥외 소화전 앞 불법주차금지입니다.")
        else:
            if access == "출입 가능":
                messages.append(f"{lp_spaced} 번님 금일 주차가능입니다.")
            elif access == "출입 불가능":
                messages.append(f"{lp_spaced} 번님 금일 주차 불가능입니다.")

    for i, message in enumerate(messages):
        audio = tts_model.infer(message, 0)
        output_audio_path = os.path.join(args.output_dir, f'user_tts_output_{i}.wav')
        tts_model.save_audio(audio, output_audio_path)
        print(f"TTS output saved to {output_audio_path}")

        if os.name == 'nt':
            subprocess.run(['powershell', '-c', f'(New-Object Media.SoundPlayer "{output_audio_path}").PlaySync();'])
        else:
            subprocess.run(['aplay', output_audio_path])

if __name__ == '__main__':
    args = get_args()
    main(args)

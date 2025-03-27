import os
import random
import wave
import librosa
import soundfile as sf

# KSS 데이터셋 경로
dataset_path = 'archive/kss'
transcript_file = 'archive/transcript.v.1.4.txt'

# 목표 샘플링 레이트
target_sr = 22050

# 파일 경로가 올바른지 확인
if not os.path.exists(transcript_file):
    print(f"파일이 존재하지 않습니다: {transcript_file}")
    exit(1)  # 프로그램 종료
else:
    print(f"파일이 존재합니다: {transcript_file}")

# 데이터 분할 비율
train_ratio = 0.8

# 파일 리스트 작성
filelist = []

# 텍스트 파일 읽기
with open(transcript_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

def check_wav_sample_rate(file_path):
    with wave.open(file_path, 'r') as wav_file:
        sample_rate = wav_file.getframerate()
    return sample_rate

def resample_wav(input_path, target_sr):
    y, sr = librosa.load(input_path, sr=None)
    if sr != target_sr:
        y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sf.write(input_path, y_resampled, target_sr)
        print(f"Resampled {input_path} from {sr} to {target_sr}")

for line in lines:
    parts = line.strip().split('|')
    if len(parts) < 2:
        continue  # 잘못된 형식의 라인 무시
    wav_path, text = parts[0], parts[1]
    full_wav_path = os.path.relpath(os.path.join(dataset_path, wav_path.replace('\\', '/')), start=os.path.dirname(transcript_file)).replace('\\', '/')
    
    # 샘플링 레이트 확인 및 변환
    if check_wav_sample_rate(full_wav_path) != target_sr:
        resample_wav(full_wav_path, target_sr)
    
    filelist.append(f"{full_wav_path}|{text}")

# 데이터 셔플
random.shuffle(filelist)

# Train/Val 분할
train_size = int(len(filelist) * train_ratio)
train_filelist = filelist[:train_size]
val_filelist = filelist[train_size:]

# 파일 저장 경로 설정
train_output_file = 'data/filelist_train.txt'
val_output_file = 'data/filelist_val.txt'

# 디렉토리 생성
os.makedirs(os.path.dirname(train_output_file), exist_ok=True)

# Train 파일 저장
with open(train_output_file, 'w', encoding='utf-8') as f:
    for line in train_filelist:
        f.write(line + '\n')

# Val 파일 저장
with open(val_output_file, 'w', encoding='utf-8') as f:
    for line in val_filelist:
        f.write(line + '\n')

print(f"Train 데이터 파일 리스트가 {train_output_file}에 작성되었습니다.")
print(f"Validation 데이터 파일 리스트가 {val_output_file}에 작성되었습니다.")
import os
import pandas as pd
import torch
import IPython.display as ipd
from scipy.io.wavfile import write

import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

class vits():
    def __init__(self, checkpoint_path, config_path):
        self.hps = utils.get_hparams_from_file(config_path)
        self.spk_count = self.hps.data.n_speakers
        self.net_g = SynthesizerTrn(
            len(symbols),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model).cuda()
        _ = self.net_g.eval()
        _ = utils.load_checkpoint(checkpoint_path, self.net_g, None)

    def get_text(self, text, hps):
        text_norm = text_to_sequence(text, hps.data.text_cleaners)
        if hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def infer(self, text, spk_id=0):
        ipd.clear_output()
        stn_tst = self.get_text(text, self.hps)
        with torch.no_grad():
            x_tst = stn_tst.cuda().unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
            sid = torch.LongTensor([spk_id]).cuda()
            audio = self.net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
        return audio

    def save_audio(self, audio, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        write(filename, self.hps.data.sampling_rate, audio)
        ipd.display(ipd.Audio(audio, rate=self.hps.data.sampling_rate, normalize=False))

# Change to the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Initialize TTS
checkpoint_path = 'checkpoints/lasttry/G_51000.pth'
config_path = 'checkpoints/lasttry/config.json'
tts = vits(checkpoint_path, config_path)

# Load CSV file
csv_file_path = os.path.join(script_dir, '../output', 'result.csv')
df = pd.read_csv(csv_file_path)

# 열 이름과 데이터 타입 확인
print(df.columns)
print(df.dtypes)

# Generate TTS messages based on conditions
messages = []

def add_space_to_license_plate(plate):
    return '  '.join(list(plate))

for index, row in df.iterrows():
    license_plate = add_space_to_license_plate(row["차량 번호"])
    fire_hydrant_detected = row["소화전 탐지"] == "탐지"
    access_result = row["출입 가능 여부"]

    # 소화전 탐지 여부를 우선적으로 확인하여 메시지를 생성
    if fire_hydrant_detected:
        messages.append(f"{license_plate} 번님 옥외 소화전 앞 불법주차금지입니다.")
    else:
        if access_result == "출입 가능":
            messages.append(f"{license_plate} 번님 금일 주차가능입니다.")
        elif access_result == "출입 불가능":
            messages.append(f"{license_plate} 번님 금일 주차 불가능입니다.")

# Convert messages to speech and save as audio files
for i, message in enumerate(messages):
    audio = tts.infer(message, 0)
    output_audio_path = os.path.join(script_dir, f'../output/user_tts_output_{i}.wav')
    tts.save_audio(audio, output_audio_path)
    print(f"TTS output saved to {output_audio_path}")

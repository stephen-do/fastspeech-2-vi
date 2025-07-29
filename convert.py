

import os
import librosa
import soundfile as sf

input_dir = 'output/result/vpb/'      # Thư mục chứa các file gốc
output_dir = 'output/result/vpb_8k/' # Thư mục để lưu file đã convert
target_sr = 8000                # Sample rate mục tiêu

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith('.wav'):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # Load file với sample rate gốc
        y, sr = librosa.load(input_path, sr=None)

        # Resample về 8kHz nếu cần
        if sr != target_sr:
            y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        else:
            y_resampled = y

        # Ghi file mới
        sf.write(output_path, y_resampled, target_sr)
        print(f"✅ Converted {filename} ({sr} Hz → {target_sr} Hz)")

print("🎉 Hoàn tất chuyển đổi tất cả file về 8kHz.")

import os
import shutil

# Thư mục chứa file lab/wav
src_dir = "../linh/trimmed_wavs"
# Thư mục đích
dst_dir = "raw_data/speaker1_linh_record_new"

# Tạo thư mục đích nếu chưa có
os.makedirs(dst_dir, exist_ok=True)

keyword = "xin cảm ơn"

for root, _, files in os.walk(src_dir):
    for file in files:
        if file.endswith(".lab"):
            lab_path = os.path.join(root, file)
            wav_path = os.path.join(root, file.replace(".lab", ".wav"))

            # Đọc nội dung file lab
            with open(lab_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().lower()

            if keyword in content:
                # Copy file lab
                shutil.copy2(lab_path, dst_dir)

                # Copy file wav nếu có
                if os.path.exists(wav_path):
                    shutil.copy2(wav_path, dst_dir)

print("✅ Đã copy xong các file chứa 'xin cảm ơn'")

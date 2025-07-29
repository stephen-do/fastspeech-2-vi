# # import onnxruntime
# # import torch
# # import numpy as np

# # session = onnxruntime.InferenceSession("FastSpeech_2.onnx", providers=['CPUExecutionProvider'])

# # texts_len = 100
# # speakers = np.array([0], dtype=np.int64)
# # texts = np.random.randint(1, 200, size=(1, texts_len)).astype(np.int64)
# # src_lens = np.array([texts_len], dtype=np.int64)
# # max_src_len = np.array(texts_len, dtype=np.int64)  # scalar

# # p_control = np.array([1], dtype=np.double)
# # e_control = np.array([1], dtype=np.double)
# # d_control = np.array([1], dtype=np.double)

# # # Prepare input dictionary
# # inputs = {
# #     # 'speakers': speakers,
# #     'texts': texts,
# #     'src_lens.1': src_lens,
# #     'max_src_len': max_src_len,
# #     "p_control": np.array(1.0, dtype=np.float64),
# #     'd_control': np.array(1.0, dtype=np.float64),
# # }

# # for input in session.get_inputs():
# #     print(f"Input name: {input.name}, shape: {input.shape}, type: {input.type}")

# # # Run inference
# # import time
# # for i in range(2):
# #     start = time.time()
# #     outputs = session.run(None, inputs)
# #     end = time.time()
# #     print('time: ', end - start)

# # # Output names
# # output_names = [
# #     'output', 'postnet_output', 'p_predictions', 'e_predictions',
# #     'log_d_predictions', 'd_rounded', 'src_masks', 'mel_masks',
# #     'src_lens', 'mel_lens'
# # ]

# # # Print result shapes
# # for name, out in zip(output_names, outputs):
# #     print(f"{name}: shape = {out.shape}")

# import re
# import argparse
# import numpy as np
# import yaml
# import onnxruntime

# from string import punctuation
# from pypinyin import pinyin, Style
# from g2p_en import G2p
# from text import text_to_sequence, clean_vietnamese_text
# import text.vietnamese_phonemes as viphonemes
# import torch
# from utils.model import get_vocoder, vocoder_infer  # Giả sử bạn có hàm load vocoder
# import soundfile as sf
# from text.symbols import get_symbols


# def read_lexicon(path):
#     lex = {}
#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             parts = re.split(r"\s+", line.strip())
#             lex[parts[0].lower()] = parts[1:]
#     return lex


# def preprocess_text(text, lang, config):
#     lexicon = read_lexicon(config["path"]["lexicon_path"])
#     text = re.sub(r"[,:;!?]", " <sp> <sp> ", clean_vietnamese_text(text))
#     words = re.split(r"([,;.\-\?\!\s+])", text)
#     phones = [ph for w in words if w.strip() for ph in (lexicon.get(w) or viphonemes.parse_word(w))]
#     phoneme_seq = "{" + " ".join(phones) + "}"
#     print("Raw Text:", text)
#     print("Phones:", phoneme_seq)
#     return np.array(text_to_sequence(phoneme_seq, config["preprocessing"]["text"]["text_cleaners"]))


# def infer(session, speakers, texts, src_lens, max_src_len, p, e, d):
#     return session.run(None, {
#         # "speakers": speakers.astype(np.int64),
#         "texts": texts.astype(np.int64),
#         "src_lens.1": src_lens.astype(np.int64),
#         "max_src_len": max_src_len.astype(np.int64),
#         "p_control": np.array([p], dtype=np.float64),
#         "e_control": np.array([e], dtype=np.float64),
#         "d_control": np.array([d], dtype=np.float64),
#     })

# def pad_sequence(seq, max_len, pad_value):
#     padded = np.pad(seq, (0, max_len - len(seq)), mode="constant", constant_values=pad_value)
#     return padded



# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--onnx_model", type=str, default='FastSpeech_2.onnx')
#     parser.add_argument("--text", type=str, required=True)
#     parser.add_argument("--speaker_id", type=int, default=0)
#     parser.add_argument("--pitch_control", type=float, default=1.0)
#     parser.add_argument("--energy_control", type=float, default=1.0)
#     parser.add_argument("--duration_control", type=float, default=1.0)
#     parser.add_argument("--preprocess_config", type=str, default='config/vpb/preprocess.yaml')
#     args = parser.parse_args()

#     config = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
#     lang = config["preprocessing"]["text"]["language"]
#     # Load vocoder
#     model_config = yaml.load(open('config/vpb/model.yaml', "r"), Loader=yaml.FullLoader)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     vocoder = get_vocoder(model_config, device) 

#     vi_symbols = get_symbols(vi_lang=True)
#     sil_token = "@sp"
#     sil_id = vi_symbols.index(sil_token)

#     seq = preprocess_text(args.text, lang, config)
#     # padded_seq = pad_sequence(seq, 200, sil_id)
#     len_remain = 200 - len(seq)
#     padded_seq = seq

#     speakers = np.array([args.speaker_id])
#     texts = np.array([padded_seq])
#     src_lens = np.array([len(seq)])
#     max_src_len = np.array(200, dtype=np.int64)

#     session = onnxruntime.InferenceSession(args.onnx_model, providers=["CPUExecutionProvider"])
    
#     for input in session.get_inputs():
#         print(f"Input name: {input.name}, shape: {input.shape}, type: {input.type}")

#     import time
#     start = time.time()
#     outputs = infer(session, speakers, texts, src_lens, max_src_len, args.pitch_control, args.energy_control, args.duration_control)
#     mel = outputs[1]
#     mel_output = outputs[1]          # shape: (1, T_max, 80)
#     mel_output = mel_output[:, :len_remain, :]  # shape: (1, T_real, 80)
#     mel_tensor = torch.from_numpy(mel_output).to(torch.float32).to(device)
#     mel_tensor = mel_tensor.transpose(1, 2)
#     wav = vocoder_infer(mel_tensor, vocoder, model_config, config)[0]
#     end = time.time()
#     print(end - start)
#     sf.write("output.wav", wav, samplerate=22050)
import onnx
import onnxruntime as ort
import numpy as np

# Load ONNX model
onnx_model = onnx.load("FastSpeech_2.onnx")
onnx.checker.check_model(onnx_model)  # Check model validity

# Create inference session
session = ort.InferenceSession("FastSpeech_2.onnx")

# Prepare inputs
inputs = {
    "texts": np.random.randint(1, 200, (1, 200), dtype=np.int64),
    "src_lens.1": np.array([100], dtype=np.int64),
    "max_src_len": np.array(200, dtype=np.int64),
    "p_control": np.array(1.0, dtype=np.float32),
    "e_control": np.array(1.0, dtype=np.float32),
    "d_control": np.array(1.0, dtype=np.float32),
}

# Run inference
outputs = session.run(None, inputs)
print([output.shape for output in outputs])  # Check output shapes
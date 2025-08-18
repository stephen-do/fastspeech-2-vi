import re
import argparse
import numpy as np
import yaml
import onnxruntime
from text import text_to_sequence, clean_vietnamese_text
import text.vietnamese_phonemes as viphonemes
import torch
from utils.model import get_vocoder, vocoder_infer
from scipy.io import wavfile


def read_lexicon(path):
    lex = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = re.split(r"\s+", line.strip())
            lex[parts[0].lower()] = parts[1:]
    return lex


def preprocess_text(text, lang, config):
    lexicon = read_lexicon(config["path"]["lexicon_path"])
    text = re.sub(r"[,:;!?]", " <sp> <sp> ", clean_vietnamese_text(text))
    words = re.split(r"([,;.\-\?\!\s+])", text)
    phones = [ph for w in words if w.strip() for ph in (lexicon.get(w) or viphonemes.parse_word(w))]
    phoneme_seq = "{" + " ".join(phones) + "}"
    print("Raw Text:", text)
    print("Phones:", phoneme_seq)
    return np.array(text_to_sequence(phoneme_seq, config["preprocessing"]["text"]["text_cleaners"]))


def infer(session, seq, p_control, e_control, d_control):
    texts = np.array([seq]).astype(np.int64)
    src_lens = np.array([len(seq)]).astype(np.int64)
    max_src_len = np.array(texts.shape[1], dtype=np.int64)  # scalar
    p_control = np.array(p_control, dtype=np.float32)
    e_control = np.array(e_control, dtype=np.float32)
    d_control = np.array(d_control, dtype=np.float32)
    return session.run(
        None,
        {
            "texts": texts,
            "src_lens.1": src_lens,
            "max_src_len": max_src_len,
            "p_control": p_control,
            "e_control": e_control,
            "d_control": d_control,
        }
    )

def pad_sequence(seq, max_len, pad_value):
    padded = np.pad(seq, (0, max_len - len(seq)), mode="constant", constant_values=pad_value)
    return padded



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_model", type=str, default='FastSpeech_2.onnx')
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--speaker_id", type=int, default=0)
    parser.add_argument("--pitch_control", type=float, default=1.0)
    parser.add_argument("--energy_control", type=float, default=1.0)
    parser.add_argument("--duration_control", type=float, default=0.85)
    args = parser.parse_args()
    preprocess_config = yaml.load(open('config/vpb/preprocess.yaml', "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open('config/vpb/model.yaml', "r"), Loader=yaml.FullLoader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocoder = get_vocoder(model_config, device)

    seq = preprocess_text(args.text, preprocess_config["preprocessing"]["text"]["language"], preprocess_config)

    session = onnxruntime.InferenceSession(args.onnx_model, providers=["CUDAExecutionProvider"])
    outputs = infer(session, seq, args.pitch_control, args.energy_control, args.duration_control)
    mel_tensor = torch.from_numpy(outputs[1]).to(torch.float32).to(device).transpose(1, 2)

    lengths = outputs[9] * preprocess_config["preprocessing"]["stft"]["hop_length"]
    wav = vocoder_infer(
        mel_tensor, vocoder, model_config, preprocess_config, lengths=lengths
    )
    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    wavfile.write("output.wav", sampling_rate, wav[0])

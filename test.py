import re
import argparse
from string import punctuation

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from g2p_en import G2p
from pypinyin import pinyin, Style
import librosa
import scipy.signal

from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples
from dataset import TextDataset
from text import text_to_sequence, clean_vietnamese_text
import text.vietnamese_phonemes as viphonemes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SpectralMatcher:
    """Real-time spectral matching post-processor for voice consistency"""
    
    def __init__(self, reference_audio_path=None, sr=22050, n_fft=1024, hop_length=256):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.reference_envelope = None
        self.reference_f0_stats = None
        
        if reference_audio_path:
            self.load_reference(reference_audio_path)
    
    def load_reference(self, audio_path):
        """Load và analyze reference audio"""
        ref_audio, _ = librosa.load(audio_path, sr=self.sr)
        
        # Extract spectral envelope
        stft = librosa.stft(ref_audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        self.reference_envelope = np.mean(magnitude, axis=1, keepdims=True)
        
        # Extract F0 statistics
        f0, voiced_flag, voiced_probs = librosa.pyin(
            ref_audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
        )
        f0_clean = f0[voiced_flag]
        if len(f0_clean) > 0:
            self.reference_f0_stats = {
                'mean': np.mean(f0_clean),
                'std': np.std(f0_clean),
                'median': np.median(f0_clean)
            }
        print(f"Reference loaded: F0 mean={self.reference_f0_stats['mean']:.1f}Hz")
    
    def apply_spectral_matching(self, audio):
        """Apply spectral envelope matching với low latency"""
        if self.reference_envelope is None:
            return audio
            
        # STFT
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Spectral envelope matching
        current_envelope = np.mean(magnitude, axis=1, keepdims=True)
        # Smooth transfer function để tránh artifacts
        transfer_func = self.reference_envelope / (current_envelope + 1e-8)
        transfer_func = scipy.signal.savgol_filter(transfer_func.flatten(), 11, 3)
        transfer_func = transfer_func.reshape(-1, 1)
        
        # Apply với gentle blending
        alpha = 0.7  # Blending factor
        modified_magnitude = magnitude * (alpha * transfer_func + (1 - alpha))
        
        # Reconstruct
        modified_stft = modified_magnitude * np.exp(1j * phase)
        modified_audio = librosa.istft(modified_stft, hop_length=self.hop_length)
        
        return modified_audio
    
    def apply_f0_normalization(self, audio):
        """Normalize F0 về reference statistics"""
        if self.reference_f0_stats is None:
            return audio
            
        # Extract F0
        f0, voiced_flag, _ = librosa.pyin(
            audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
        )
        
        if np.sum(voiced_flag) == 0:
            return audio
            
        # Calculate scaling factor
        f0_voiced = f0[voiced_flag]
        current_median = np.median(f0_voiced)
        scale_factor = self.reference_f0_stats['median'] / current_median
        
        # Apply gentle F0 scaling với PSOLA-like approach
        if 0.8 <= scale_factor <= 1.2:  # Only apply moderate corrections
            # Simple time-domain pitch scaling
            if scale_factor != 1.0:
                audio = librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=12*np.log2(scale_factor))
        
        return audio
    
    def process(self, audio):
        """Main processing pipeline"""
        # Apply F0 normalization first
        audio = self.apply_f0_normalization(audio)
        # Then spectral matching
        audio = self.apply_spectral_matching(audio)
        return audio


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path, 'r', encoding='utf-8') as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_vietnamese(text, preprocess_config):
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])
    text = text.replace(',', ' ').replace('.', ' ').replace(';', ' ').replace('?', ' ').replace('!', ' ').replace(':', ' ')
    text = text + ' <sp> <sp> <sp> <sp> <sp> <sp> <sp> <sp> '
    text = clean_vietnamese_text(text)
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w in lexicon:
            phones += lexicon[w]
        elif w!=' ':
            phones += viphonemes.parse_word(w)

    phones = "{" + " ".join(phones) + "}"
    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)


def synthesize(model, step, configs, vocoder, batchs, control_values, spectral_matcher=None):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            output = model(
                *(batch[2:]),
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control
            )

            # Modified synth_samples với spectral matching
            synth_samples_with_matching(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
                spectral_matcher
            )


def synth_samples_with_matching(batch, output, vocoder, model_config, preprocess_config, 
                               result_path, spectral_matcher=None):
    """Modified synth_samples với spectral matching post-processing"""
    import os
    import soundfile as sf
    
    # Extract mel-spectrograms từ model output
    mel_predictions = output[1]  # Assuming output[1] contains mel predictions
    
    for i in range(len(mel_predictions)):
        basename = batch[0][i]  # filename
        mel = mel_predictions[i].cpu().numpy()
        
        # Vocoder synthesis
        with torch.no_grad():
            mel_torch = torch.from_numpy(mel).unsqueeze(0).to(device)
            audio = vocoder(mel_torch).squeeze().cpu().numpy()
        
        # Apply spectral matching nếu có
        if spectral_matcher is not None:
            print(f"Applying spectral matching to {basename}...")
            audio = spectral_matcher.process(audio)
        
        # Save audio
        audio_path = os.path.join(result_path, f"{basename}.wav")
        sf.write(audio_path, audio, preprocess_config["preprocessing"]["audio"]["sampling_rate"])
        print(f"Generated: {audio_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default='200000')
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single"],
        default='single',
        help="Synthesize a whole dataset or a single sentence",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="path to a source file with format like train.txt and val.txt, for batch mode only",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="raw text to synthesize, for single-sentence mode only",
    )
    parser.add_argument(
        "--speaker_id",
        type=int,
        default=0,
        help="speaker ID for multi-speaker synthesis, for single-sentence mode only",
    )
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        default='config/vpb/preprocess.yaml',
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, default='config/vpb/model.yaml', help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, default='config/vpb/train.yaml', help="path to train.yaml"
    )
    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.0,
        help="control the pitch of the whole utterance, larger value for higher pitch",
    )
    parser.add_argument(
        "--energy_control",
        type=float,
        default=1.0,
        help="control the energy of the whole utterance, larger value for larger volume",
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )
    # New argument cho spectral matching
    parser.add_argument(
        "--reference_audio",
        type=str,
        default='/hdd_data01/tuyendn3/eda_aic_tts/raw_data/speaker1_linh_record_new/linhlnn1_0024.wav',
        help="path to reference audio for spectral matching consistency",
    )
    args = parser.parse_args()

    # Check source texts
    if args.mode == "batch":
        assert args.source is not None and args.text is None
    if args.mode == "single":
        assert args.source is None and args.text is not None
        
    preprocess_config = yaml.load(
            open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(args, configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)
    
    # Initialize spectral matcher
    spectral_matcher = None
    if args.reference_audio:
        print(f"Loading reference audio: {args.reference_audio}")
        spectral_matcher = SpectralMatcher(
            reference_audio_path=args.reference_audio,
            sr=preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        )

    # Preprocess texts
    if args.mode == "batch":
        # Get dataset
        dataset = TextDataset(args.source, preprocess_config)
        batchs = DataLoader(
                dataset,
                batch_size=8,
                collate_fn=dataset.collate_fn,
            )
    if args.mode == "single":
        ids = raw_texts = [args.text[:100]]
        speakers = np.array([args.speaker_id])
        texts = np.array([preprocess_vietnamese(args.text, preprocess_config)])
        text_lens = np.array([len(texts[0])])
        batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]

    control_values = args.pitch_control, args.energy_control, args.duration_control
        
    synthesize(model, args.restore_step, configs, vocoder, batchs, control_values, spectral_matcher)
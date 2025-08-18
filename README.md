# Apply FastSpeech 2 model to Vietnamese TTS

## Dataset
- [Infore](https://huggingface.co/datasets/ntt123/infore/resolve/main/infore_16k_denoised.zip): a single speaker Vietnamese dataset with 14935 short audio clips of a female speaker
- Download and extract files into ``./raw_data/infore/``

## Montreal Forced Aligner
- Recommended version: 2.0.6

## Preprocess data and train model 
- Do step by step according to scripts included in ``./scripts/infore/``
- Alignment of the dataset Infore at here [Infore's alignment](https://drive.google.com/file/d/1pDlwYDn2xW2_nnW5dzyVecmb_xbMsaQx/view?usp=sharing) : download and extract into ``./proprocessed_data/infore/``

## Pretrained model
- Download and extract [pretrained model](https://drive.google.com/file/d/1R0RuE75JlPR2_ApsTrk6z4rPhgLKvK-B/view?usp=sharing) into ``./output/ckpt/infore/``

## Inferrence
```
python3 synthesize.py --text "YOUR_DESIRED_TEXT" --restore_step 100000 --mode single -p config/infore/preprocess.yaml -m config/infore/model.yaml -t config/infore/train.yaml
```

# References
- [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558), Y. Ren, *et al*.
- [xcmyz's FastSpeech implementation](https://github.com/xcmyz/FastSpeech)
- [TensorSpeech's FastSpeech 2 implementation](https://github.com/TensorSpeech/TensorflowTTS)
- [rishikksh20's FastSpeech 2 implementation](https://github.com/rishikksh20/FastSpeech2)

# Citation
```
@INPROCEEDINGS{chien2021investigating,
  author={Chien, Chung-Ming and Lin, Jheng-Hao and Huang, Chien-yu and Hsu, Po-chun and Lee, Hung-yi},
  booktitle={ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Investigating on Incorporating Pretrained and Learnable Speaker Representations for Multi-Speaker Multi-Style Text-to-Speech}, 
  year={2021},
  volume={},
  number={},
  pages={8588-8592},
  doi={10.1109/ICASSP39728.2021.9413880}}
```

import torch
import yaml
import argparse
from utils.model import get_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=200000)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define input and output names
    input_names = ["speakers", "texts", "src_lens", "max_src_len", "p_control", "e_control", "d_control"]
    output_names = [
        "output", "postnet_output", "p_predictions", "e_predictions",
        "log_d_predictions", "d_rounded", "src_masks", "mel_masks",
        "src_lens", "mel_lens"
    ]
    # Define dynamic axes
    dynamic_axes = {
        "speakers": {0: "batch_size"},
        "texts": {0: "batch_size", 1: "text_lens"},   # batch, seq_len
        "src_lens": {0: "batch_size"},
        "max_src_len": {},
        "output": {0: "batch_size", 1: "mel_len"},
        "postnet_output": {0: "batch_size", 1: "mel_len"},
        "p_predictions": {0: "batch_size", 1: "text_lens"},
        "e_predictions": {0: "batch_size", 1: "text_lens"},
        "log_d_predictions": {0: "batch_size", 1: "text_lens"},
        "d_rounded": {0: "batch_size", 1: "text_lens"},
        "src_masks": {0: "batch_size", 1: "text_lens"},
        "mel_masks": {0: "batch_size", 1: "mel_len"}
    }

    # Dummy inputs
    texts_len = 200
    speakers = torch.tensor([0], dtype=torch.int64).to(device)
    texts = torch.randint(1, 200, (1, texts_len), dtype=torch.int64).to(device)
    text_lens = torch.tensor([texts_len], dtype=torch.int64).to(device)
    max_len = torch.tensor(200, dtype=torch.int64).to(device)
    p_control = torch.tensor(1.0, dtype=torch.float32).to(device)
    e_control = torch.tensor(1.0, dtype=torch.float32).to(device)
    d_control = torch.tensor(1.0, dtype=torch.float32).to(device)

    # Load configs
    preprocess_config = yaml.load(open('config/vpb/preprocess.yaml', "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open('config/vpb/model.yaml', "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open('config/vpb/train.yaml', "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Load model
    model = get_model(args, configs, device, train=False)
    model.eval()  # Set model to evaluation mode

    # Export to ONNX
    torch.onnx.export(
        model,
        args=(
            speakers, texts, text_lens, max_len,
            None, None, None,  # mels, mel_lens, max_mel_len
            None, None, None,  # p_targets, e_targets, d_targets
            p_control, e_control, d_control
        ),
        f="./FastSpeech_2.onnx",
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        verbose=True  # Use verbose instead of report
    )

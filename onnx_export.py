# import torch
# import numpy as np

# import torch
# import yaml
# import numpy as np
# import argparse
# from utils.model import get_model


# if __name__ == "__main__":

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--restore_step", type=int, default=157000, required=False)
#     args = parser.parse_args()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     input_names=[
#             "speakers", "texts", "src_lens", "max_src_len",
#             "p_control", "e_control", "d_control"]
#     output_names = ['output', 'postnet_output', 'p_predictions', 'e_predictions', 'log_d_predictions', 'd_rounded',
#                    'src_masks', 'mel_masks', 'src_lens', 'mel_lens']
#     dynamic_axes = {
#             "texts": {0: "texts_len"},
#             "output": {1: "output_len"},
#             "postnet_output": {1: "postnet_output_len"},
#             "p_predictions": {1: "p_predictions_len"},
#             "e_predictions": {1: "e_predictions_len"},
#             "log_d_predictions": {1: "log_d_predictions_len"},
#             "d_rounded": {1: "d_rounded_len"},
#             "src_masks": {1: "src_masks_len"}
#     }
# #     dynamic_axes = {
# #         "texts": {0: "batch_size", 1: "text_lens"},
# #         "src_lens": {0: "batch_size"},
# #         # If you want max_src_len to be dynamic, add here
# #         "output": {0: "batch_size", 1: "mel_len"},
# #         "postnet_output": {0: "batch_size", 1: "mel_len"},
# #         # Add dynamic axes for other outputs as needed
# #     }

#     texts_len = 200
#     speakers = torch.tensor([0]).to(device)
#     texts = torch.randint(1, 200, (1, texts_len)).to(device)  
#     text_lens = torch.tensor([texts_len]).to(device)
#     max_len = torch.from_numpy(np.array(200)).to(device)
#     p_control = torch.tensor(1.0, dtype=torch.float64).to(device)
#     d_control = torch.tensor(1.0, dtype=torch.float64).to(device)
#     e_control = torch.tensor(1.0, dtype=torch.float64).to(device)

#     preprocess_config = yaml.load(open('config/vpb/preprocess.yaml', "r"), Loader=yaml.FullLoader)
#     model_config = yaml.load(open('config/vpb/model.yaml', "r"), Loader=yaml.FullLoader)
#     train_config = yaml.load(open('config/vpb/train.yaml', "r"), Loader=yaml.FullLoader)
#     configs = (preprocess_config, model_config, train_config)

#     model = get_model(args, configs, device, train=False)


#     torch.onnx.export(model, args=(
#                                 speakers,
#                                 texts,
#                                 text_lens,
#                                 max_len,  # max_src_len
#                                 None, None, None,  # mels, mel_lens, max_mel_len
#                                 None, None, None,  # p_targets, e_targets, d_targets
#                                 p_control,
#                                 e_control,
#                                 d_control
#                         ), f="./FastSpeech_2.onnx",
#                         input_names=input_names, output_names=output_names, 
#                         dynamic_axes=dynamic_axes, opset_version=17,
#                         report=True)
# #     onnx_program  = torch.onnx.export(model, args=(
# #                                 speakers,
# #                                 texts,
# #                                 text_lens,
# #                                 max_len,  # max_src_len
# #                                 None, None, None,  # mels, mel_lens, max_mel_len
# #                                 None, None, None,  # p_targets, e_targets, d_targets
# #                                 p_control,
# #                                 e_control,
# #                                 d_control
# #                         ),dynamo=True, opset_version=17)
# #     onnx_program.optimize()
# #     onnx_program.save("mlp.onnx")

import torch
import yaml
import argparse
from utils.model import get_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=157000)
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
        "texts": {0: "batch_size", 1: "text_lens"},
        "src_lens": {0: "batch_size"},
        "max_src_len": {},  # Scalar, but allow dynamic if needed
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
    
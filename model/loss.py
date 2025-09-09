import torch
import torch.nn as nn
from pytorch_msssim import ssim

class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2Loss, self).__init__()
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"]["feature"]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"]["feature"]
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

        # Hệ số alpha: 0.9 MAE + 0.1 SSIM
        self.alpha = 0.6

    def compute_mel_loss_with_ssim(self, pred, target, mask):
        """
        Kết hợp MAE và SSIM loss
        Input shape: [B, T, D], mask shape: [B, T]
        """

        # L1 Loss (MAE)
        l1_loss = self.mae_loss(pred.masked_select(mask.unsqueeze(-1)), target.masked_select(mask.unsqueeze(-1)))

        # Chuẩn hóa về [0, 1] để dùng SSIM
        pred = pred.detach()  # Không ảnh hưởng gradient
        target = target.detach()

        pred_min, pred_max = pred.min(), pred.max()
        target_min, target_max = target.min(), target.max()

        pred_norm = (pred - pred_min) / (pred_max - pred_min + 1e-5)
        target_norm = (target - target_min) / (target_max - target_min + 1e-5)

        # Reshape [B, T, D] → [B, 1, D, T]
        pred_img = pred_norm.transpose(1, 2).unsqueeze(1)
        target_img = target_norm.transpose(1, 2).unsqueeze(1)

        # SSIM loss
        ssim_val = ssim(pred_img, target_img, data_range=1.0, size_average=True)
        ssim_loss = 1 - ssim_val

        return self.alpha * l1_loss + (1 - self.alpha) * ssim_loss

    def forward(self, inputs, predictions):
        (
            mel_targets,
            _,
            _,
            pitch_targets,
            energy_targets,
            duration_targets,
        ) = inputs[6:]
        (
            mel_predictions,
            postnet_mel_predictions,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            _,
            src_masks,
            mel_masks,
            _,
            _,
        ) = predictions
        src_masks = ~src_masks
        mel_masks = ~mel_masks
        log_duration_targets = torch.log(duration_targets.float() + 1)
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False

        if self.pitch_feature_level == "phoneme_level":
            pitch_predictions = pitch_predictions.masked_select(src_masks)
            pitch_targets = pitch_targets.masked_select(src_masks)
        elif self.pitch_feature_level == "frame_level":
            pitch_predictions = pitch_predictions.masked_select(mel_masks)
            pitch_targets = pitch_targets.masked_select(mel_masks)

        if self.energy_feature_level == "phoneme_level":
            energy_predictions = energy_predictions.masked_select(src_masks)
            energy_targets = energy_targets.masked_select(src_masks)
        if self.energy_feature_level == "frame_level":
            energy_predictions = energy_predictions.masked_select(mel_masks)
            energy_targets = energy_targets.masked_select(mel_masks)

        log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)

        # Tính loss kết hợp MAE + SSIM
        mel_loss = self.compute_mel_loss_with_ssim(mel_predictions.float(), mel_targets.float(), mel_masks)
        postnet_mel_loss = self.compute_mel_loss_with_ssim(postnet_mel_predictions.float(), mel_targets.float(), mel_masks)

        pitch_loss = self.mse_loss(pitch_predictions.float(), pitch_targets.float())
        energy_loss = self.mse_loss(energy_predictions.float(), energy_targets.float())
        duration_loss = self.mse_loss(log_duration_predictions.float(), log_duration_targets.float())

        total_loss = mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss

        return (
            total_loss,
            mel_loss,
            postnet_mel_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
        )

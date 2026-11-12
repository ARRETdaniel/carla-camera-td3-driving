"""CNN Feature Extractor for End-to-End Autonomous Vehicle Navigation."""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Tuple


class NatureCNN(nn.Module):

    def __init__(
        self,
        input_channels: int = 4,
        num_frames: int = 4,
        feature_dim: int = 512,
    ):
        super(NatureCNN, self).__init__()

        self.input_channels = input_channels
        self.num_frames = num_frames
        self.feature_dim = feature_dim

        self.logger = logging.getLogger(__name__)

        self.forward_step_counter = 0
        self.log_frequency = 100

        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=32,
            kernel_size=8,
            stride=4,
            padding=0,
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=0,
        )

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=0,
        )

        self.ln1 = nn.LayerNorm([32, 20, 20])
        self.ln2 = nn.LayerNorm([64, 9, 9])
        self.ln3 = nn.LayerNorm([64, 7, 7])

        self.activation = nn.LeakyReLU(negative_slope=0.01)

        self._compute_flat_size()

        self.fc = nn.Linear(self.flat_size, feature_dim)

        self.ln4 = nn.LayerNorm(feature_dim)

        self._initialize_weights()

    def _compute_flat_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_channels, 84, 84)

            out = self.activation(self.conv1(dummy_input))
            out = self.activation(self.conv2(out))
            out = self.activation(self.conv3(out))

            self.flat_size = int(np.prod(out.shape[1:]))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.forward_step_counter += 1
        should_log = (self.forward_step_counter % self.log_frequency == 0)

        if x.shape[1:] != (self.input_channels, 84, 84):
            raise ValueError(
                f"Expected input shape (batch, {self.input_channels}, 84, 84), "
                f"got {x.shape}"
            )

        if self.logger.isEnabledFor(logging.DEBUG) and should_log:
            self.logger.debug(
                f"   CNN FORWARD PASS #{self.forward_step_counter} - INPUT:\n"
                f"   Shape: {x.shape}\n"
                f"   Dtype: {x.dtype}\n"
                f"   Device: {x.device}\n"
                f"   Range: [{x.min().item():.3f}, {x.max().item():.3f}]\n"
                f"   Mean: {x.mean().item():.3f}, Std: {x.std().item():.3f}\n"
                f"   Has NaN: {torch.isnan(x).any().item()}\n"
                f"   Has Inf: {torch.isinf(x).any().item()}"
            )


        out = self.conv1(x)
        out = self.ln1(out)
        out = self.activation(out)

        if self.logger.isEnabledFor(logging.DEBUG) and should_log:
            self.logger.debug(
                f"   CNN LAYER 1 (Conv 32×8×8, stride=4 + LayerNorm):\n"
                f"   Output shape: {out.shape}\n"
                f"   Range: [{out.min().item():.3f}, {out.max().item():.3f}]\n"
                f"   Mean: {out.mean().item():.3f}, Std: {out.std().item():.3f}\n"
                f"   L2 Norm: {torch.norm(out).item():.3f}\n"
                f"   Active neurons: {(out > 0).float().mean().item()*100:.1f}%"
            )

        out = self.conv2(out)
        out = self.ln2(out)
        out = self.activation(out)

        if self.logger.isEnabledFor(logging.DEBUG) and should_log:
            self.logger.debug(
                f"   CNN LAYER 2 (Conv 64×4×4, stride=2 + LayerNorm):\n"
                f"   Output shape: {out.shape}\n"
                f"   Range: [{out.min().item():.3f}, {out.max().item():.3f}]\n"
                f"   Mean: {out.mean().item():.3f}, Std: {out.std().item():.3f}\n"
                f"   L2 Norm: {torch.norm(out).item():.3f}\n"
                f"   Active neurons: {(out > 0).float().mean().item()*100:.1f}%"
            )

        out = self.conv3(out)
        out = self.ln3(out)
        out = self.activation(out)

        if self.logger.isEnabledFor(logging.DEBUG) and should_log:
            self.logger.debug(
                f"   CNN LAYER 3 (Conv 64×3×3, stride=1 + LayerNorm):\n"
                f"   Output shape: {out.shape}\n"
                f"   Range: [{out.min().item():.3f}, {out.max().item():.3f}]\n"
                f"   Mean: {out.mean().item():.3f}, Std: {out.std().item():.3f}\n"
                f"   L2 Norm: {torch.norm(out).item():.3f}\n"
                f"   Active neurons: {(out > 0).float().mean().item()*100:.1f}%"
            )

        out = out.view(out.size(0), -1)

        features = self.fc(out)
        features = self.ln4(features)
        features = self.activation(features)

        if self.logger.isEnabledFor(logging.DEBUG) and should_log:
            self.logger.debug(
                f"   CNN FORWARD PASS - OUTPUT:\n"
                f"   Feature shape: {features.shape}\n"
                f"   Range: [{features.min().item():.3f}, {features.max().item():.3f}]\n"
                f"   Mean: {features.mean().item():.3f}, Std: {features.std().item():.3f}\n"
                f"   L2 norm: {features.norm(dim=1).mean().item():.3f}\n"
                f"   Has NaN: {torch.isnan(features).any().item()}\n"
                f"   Has Inf: {torch.isinf(features).any().item()}\n"
                f"   Feature quality: {'GOOD' if not (torch.isnan(features).any() or torch.isinf(features).any()) else 'BAD'}"
            )

        return features

    def get_feature_dim(self) -> int:
        return self.feature_dim

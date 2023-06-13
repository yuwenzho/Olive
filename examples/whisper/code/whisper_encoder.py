# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import List

import torch
from transformers import WhisperConfig


class WhisperEncoder(torch.nn.Module):
    """Whisper encoder outputs only the last hidden state"""

    def __init__(self, encoder, config: WhisperConfig):
        super().__init__()
        self.encoder = encoder
        self.config = config

    def forward(self, input_features, attention_mask):
        return self.encoder.model.encoder(input_features)[0]


class WhisperEncoderInputs:
    def __init__(self, input_features, attention_mask):
        self.input_ids: torch.LongTensor = input_features
        # HF Whisper model doesn't support Attention Mask functionality

    @staticmethod
    def create_dummy(
        batch_size: int, sequence_length: int, feature_size: int, device: torch.device, use_int32_inputs: bool
    ):
        """Create dummy inputs for Whisper encoder.

        Args:
            batch_size (int): batch size
            sequence_length (int): sequence length
            feature_size (int): feature size for spectrogram input
            device (torch.device): device of output tensors

        Returns:
            WhisperEncoderInputs: dummy inputs for encoder
        """
        dtype = torch.float32

        input_features = torch.randn(
            size=(batch_size, feature_size, sequence_length),
            device=device,
        )
        attention_mask = torch.ones([batch_size, feature_size, sequence_length], dtype=dtype, device=device)
        return WhisperEncoderInputs(input_features, attention_mask)

    def to_list(self) -> List:
        if self.input_features is None:
            return []
        return [self.input_features]
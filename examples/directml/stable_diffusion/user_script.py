# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import torch


class UNetLoader:
    def __init__(self, batchsize):
        self.batchsize = batchsize
        self.channels = 4
        self.height = 64
        self.width = 64
        self.sequence = 77

    def __getitem__(self, idx):
        input_dict = {
            "sample": torch.rand((self.batchsize, self.channels, self.height, self.width), dtype=torch.float),
            "timestep": torch.rand((self.batchsize,), dtype=torch.float),
            "encoder_hidden_states": torch.rand((self.batchsize, self.sequence, 768), dtype=torch.float),
        }
        label = None
        return input_dict, label

    def __len__(self):
        return self.size


def create_unet_dataloader(data_dir, batchsize):
    return UNetLoader(batchsize)

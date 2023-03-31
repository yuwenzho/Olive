# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import torch

class SqueezenetLoader:
    def __init__(self, batchsize):
        self.batchsize = batchsize
        self.channels = 3
        self.height = 224
        self.width = 224

    def __getitem__(self, idx):
        # input_dict = {
        #     "sample": torch.rand((self.batchsize, self.channels, self.height, self.width), dtype=torch.float),
        # }
        label = None
        return torch.rand((self.batchsize, self.channels, self.height, self.width), dtype=torch.float), label


def create_dataloader(data_dir, batchsize):
    return SqueezenetLoader(batchsize)

# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import torch


class RandomDataLoader:
    def __init__(self, batchsize):
        self.batchsize = batchsize

    def __getitem__(self, idx):
        return torch.randn(10), 1

    def __len__(self):
        return self.size


def create_unet_dataloader(data_dir, batchsize):
    return RandomDataLoader(batchsize)

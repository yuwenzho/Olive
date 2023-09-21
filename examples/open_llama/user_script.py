# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import torch
import os
from datasets import load_dataset
from transformers import AutoConfig

from olive.constants import Framework
from torch.utils.data import DataLoader
import onnxruntime as ort
from transformers import LlamaConfig, LlamaTokenizer
from torch.nn.functional import pad
import numpy as np
from olive.model import OliveModel

model_id = "openlm-research/open_llama_3b"
config = AutoConfig.from_pretrained(model_id)


class RandomDataLoader:
    def __init__(self, create_inputs_func, batch_size, torch_dtype, model_framework=Framework.PYTORCH):
        self.create_input_func = create_inputs_func
        self.batch_size = batch_size
        self.torch_dtype = torch_dtype
        self.model_framework = model_framework

    def __getitem__(self, idx):
        label = None
        return self.create_input_func(self.batch_size, self.torch_dtype, self.model_framework), label


def dummy_inputs(batch_size, torch_dtype, model_framework=Framework.PYTORCH):
    past_sequence_length = 1
    attention_mask_sequence_length = 1
    sequence_length = 2

    inputs = {
        "input_ids": torch.randint(10, (batch_size, sequence_length), dtype=torch.int64),
        "attention_mask": torch.randint(10, (batch_size, attention_mask_sequence_length), dtype=torch.int64),
    }
    rand_kv_tensor = torch.rand(
        (
            batch_size,
            config.num_attention_heads,
            past_sequence_length,
            int(config.hidden_size / config.num_attention_heads),
        ),
        dtype=torch_dtype,
    )
    if model_framework == Framework.ONNX:
        for layer_index in range(config.num_hidden_layers):
            inputs[f"past_key_values.{layer_index}.key"] = rand_kv_tensor
            inputs[f"past_key_values.{layer_index}.value"] = rand_kv_tensor
        inputs["use_cache_branch"] = torch.ones((1,), dtype=torch.bool)
    elif model_framework == Framework.PYTORCH:
        inputs["use_cache"] = True
        inputs["past_key_values"] = [torch.stack((rand_kv_tensor, rand_kv_tensor))] * config.num_hidden_layers
    return inputs


def dataloader_func(data_dir, batch_size, *args, **kwargs):
    model_framework = kwargs.get("model_framework", Framework.PYTORCH)
    return RandomDataLoader(dummy_inputs, batch_size, torch.float16, model_framework)

def tokenize_function(examples):
    tokenizer = LlamaTokenizer.from_pretrained("openlm-research/open_llama_3b")
    example = tokenizer(examples["text"])
    return example

class KVDataloader:
    def __init__(self, model_path, pad_max=196, batch_size=1, sub_folder='train'):
        self.pad_max = pad_max
        self.batch_size=batch_size
        dataset = None
        while dataset is None:
            try:
                dataset = load_dataset("NeelNanda/pile-10k", split=sub_folder)
            except:
                print('retry')
        dataset = dataset.map(tokenize_function, batched=True)
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_batch,
        )
        self.sess = None
        if not model_path.endswith('decoder_model.onnx'):
            self.sess = ort.InferenceSession(os.path.join(os.path.dirname(model_path), 'decoder_model.onnx'))


    def collate_batch(self, batch):

        input_ids_padded = []
        attention_mask_padded = []
        last_ind = []

        for text in batch:
            input_ids = text["input_ids"]
            pad_len = self.pad_max - input_ids.shape[0]
            last_ind.append(input_ids.shape[0] - 1)
            attention_mask = torch.ones(len(input_ids))
            input_ids = pad(input_ids, (0, pad_len), value=1)
            attention_mask = pad(attention_mask, (0, pad_len), value=0)
            input_ids_padded.append(input_ids)
            attention_mask_padded.append(attention_mask)
        return (torch.vstack(input_ids_padded), torch.vstack(attention_mask_padded)), torch.tensor(last_ind)


    def __iter__(self):
        try:
            for (input_ids, attention_mask), last_ind in self.dataloader:
                if self.sess is None:
                    yield {'input_ids': input_ids[:, :-1].detach().cpu().numpy().astype('int64'),
                           'attention_mask':attention_mask[:, :-1].detach().cpu().numpy().astype('int64')}, last_ind.detach().cpu().numpy()
                else:
                    outputs = self.sess.run(None, {'input_ids': input_ids[:, :-1].detach().cpu().numpy().astype('int64'),
                                                   'attention_mask':attention_mask[:, :-1].detach().cpu().numpy().astype('int64')})
                    ort_input = {}
                    ort_input['input_ids'] = input_ids[:, -1].unsqueeze(0).detach().cpu().numpy().astype('int64')
                    for i in range(int((len(outputs) - 1) / 2)):
                        ort_input['past_key_values.{}.key'.format(i)] = outputs[i*2+1]
                        ort_input['past_key_values.{}.value'.format(i)] = outputs[i*2+2]
                    ort_input['attention_mask'] =  np.zeros([self.batch_size, ort_input['past_key_values.0.key'].shape[2]+1], dtype='int64')
                    yield ort_input, last_ind.detach().cpu().numpy()
        except StopIteration:
            return

def create_onnx_dataloader(data_dir, batch_size=1, *args, **kwargs):
    model_path = kwargs.pop("model_path")
    print('model_path', model_path)
    dataloader = KVDataloader(model_path, batch_size=1)
    return dataloader

def eval_accuracy(model: OliveModel, data_dir, batch_size, device, execution_providers):
    from intel_extension_for_transformers.evaluation.lm_eval import evaluate
    import pdb;pdb.set_trace()
    
    results = evaluate(
        model="hf-causal",
        model_args="pretrained=" + model.model_path + ",tokenizer=openlm-research/open_llama_3b",
        batch_size=batch_size,
        tasks="lambada_openai",
        model_format="onnx"
    )
    print("Accuracy is: %s" % (results["results"]["lambada_openai"]["acc"]))
    return results["results"]["lambada_openai"]["acc"]

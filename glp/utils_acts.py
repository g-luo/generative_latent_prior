from baukit import TraceDict
from collections import OrderedDict
from dataclasses import dataclass
import einops
import logging
import numpy as np
import os
from pathlib import Path
import torch
from typing import Literal
from tqdm import tqdm
from tqdm import trange

from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

@torch.no_grad()
def save_acts(
    hf_model: AutoModelForCausalLM, 
    hf_tokenizer: AutoTokenizer, 
    text: list[str],
    tracedict_config: dict,
    padding_side: str = "right",
    token_idx: Literal["last", "all"] = "last",
    batch_size: int = 10,
    max_length: int = 2048
):
    # set up tracedict
    tracedict_config = dict(tracedict_config)
    retain_attr = tracedict_config.pop("retain")
    assert retain_attr in ["input", "output"], "Must retain exactly one of input or output"
    tracedict_config[f"retain_{retain_attr}"] = True
    if tracedict_config.get("layer_prefix") is not None:
        layer_prefix = tracedict_config.pop("layer_prefix")
        tracedict_config["layers"] = [f"{layer_prefix}.{layer}" for layer in tracedict_config["layers"]]
    # set up tokenizer
    if getattr(hf_tokenizer, "pad_token") is None:
        print(f"WARNING: setting tokenizer pad_token to eos_token")
        hf_tokenizer.pad_token = hf_tokenizer.eos_token
    if padding_side != hf_tokenizer.padding_side:
        print(f"WARNING: updating tokenizer padding_side to {padding_side}")
        hf_tokenizer.padding_side = padding_side
    ret = []
    for i in tqdm(range(0, len(text), batch_size)):
        start, end = i, min(i + batch_size, len(text))
        minibatch = hf_tokenizer(
            text[start:end], 
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_length
        )
        minibatch = {k: v.to(hf_model.device) for k, v in minibatch.items()}
        with TraceDict(hf_model, **tracedict_config) as miniret:
            hf_model(**minibatch)
        miniret = [getattr(miniret[l], retain_attr) for l in tracedict_config["layers"]]
        miniret = [x[0] if type(x) is tuple else x for x in miniret]
        miniret = torch.stack(miniret)
        miniret = einops.rearrange(miniret, "l b s d -> b l s d")
        if token_idx == "last":
            last_token_idx = -1 if padding_side == "left" else (minibatch["attention_mask"].sum(dim=1) - 1)
            miniret = miniret[torch.arange(miniret.shape[0]), :, last_token_idx, :].detach().cpu()
        elif token_idx == "all":
            miniret = miniret.detach().cpu()
        else:
            raise NotImplementedError
        ret.append(miniret)
    ret = torch.cat(ret, dim=0)
    return ret

@dataclass(kw_only=True)
class MemmapWriter:
    """
    Given a path path/to/dataset/, this will write to:
        path/to/dataset/data_0000.npy
        path/to/dataset/data_0001.npy
        ...
        path/to/dataset/data_indices.npy
    
    """
    output_dir: Path
    file_size: int # file size in number of elements
    dtype: np.dtype
    
    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.memmap_files = []
        self._new_memmap_file()
        self.cur_idx = 0
        self.indices: list[tuple[int, int, int]] = [] # (file_idx, start_idx, end_idx)

    def _new_memmap_file(self):
        path = self.output_dir / f'data_{len(self.memmap_files):04d}.npy'
        self.memmap_files.append(np.memmap(
                mode="w+",
                filename=path,
                dtype=self.dtype, 
                shape=self.file_size
            )
        )
        self.cur_idx = 0
        logger.info(f'Created memmap file {path} with size {self.file_size}')
        
    def write(self, chunk: np.ndarray):
        assert chunk.dtype == self.dtype
        length, = chunk.shape
        assert length <= self.file_size
        if self.cur_idx + length > self.file_size:
            self._new_memmap_file()
        self.memmap_files[-1][self.cur_idx:self.cur_idx + length] = chunk
        self.cur_idx += length
        self.indices.append((len(self.memmap_files) - 1, self.cur_idx - length, self.cur_idx))

    def flush(self):
        for memmap_file in self.memmap_files:
            memmap_file.flush()
            logger.info(f'Finished writing to {memmap_file.filename}')
        indices_path = self.output_dir / 'data_indices.npy'
        np.save(indices_path, np.array(self.indices, dtype=np.uint64))
        logger.info(f'Saved indices to {indices_path}')

@dataclass()
class MemmapReader:
    data_dir: Path
    dtype: np.dtype
    def __post_init__(self):
        indices_path = self.data_dir / 'data_indices.npy'
        self.indices = np.load(indices_path)
        logger.info(f'Loaded {len(self.indices)} indices from {indices_path}')
        # dictionary to cache open memmap files
        self._memmap_cache = OrderedDict()
        
    def __len__(self):
        return len(self.indices)
    
    def _get_memmap(self, file_idx):
        """Get or create a memmap for the given file index"""
        if file_idx not in self._memmap_cache:
            filepath = self.data_dir / f'data_{file_idx:04d}.npy'
            self._memmap_cache[file_idx] = np.memmap(
                filename=filepath,
                mode='r',
                dtype=self.dtype
            )
            if len(self._memmap_cache) > 3:
                self._memmap_cache.popitem(last=False)
        return self._memmap_cache[file_idx]
    
    def __getitem__(self, idx):
        """Get the chunk at the given index"""
        if isinstance(idx, slice):
            # handle slice indexing
            indices = range(*idx.indices(len(self)))
            return [self[i] for i in indices]
        # get the file_idx, start_idx, and end_idx for this chunk
        file_idx, start_idx, end_idx = self.indices[idx]
        # get the memmap for this file
        memmap = self._get_memmap(file_idx)
        # return the chunk
        return memmap[start_idx:end_idx]
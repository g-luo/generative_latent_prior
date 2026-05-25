from dataclasses import dataclass, field
from datasets import load_dataset
import json
import numpy as np
from omegaconf import OmegaConf
import os
import psutil
import re
import shutil
from tqdm import tqdm
from typing import Optional

import torch
from torch.utils.data import DataLoader
from nnsight.modeling.vllm import VLLM

from glp_dataset import DynamicActWriter

@dataclass
class SaveActivationsConfig:
    """
    Configuration for saving activations.
    """
    # model
    output_path: str
    model_name: str
    save_root: str = "."
    # wandb
    wandb_enabled: bool = False
    wandb_entity: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    # nnsight
    latent_model_config: dict = field(default_factory=dict)
    # vllm
    vllm_kwargs: dict = field(default_factory=dict)
    # saving
    batch_size: int = 10
    max_length: int = 2048
    acts_per_shard: int = 1000000
    max_total_acts: int = 5000000
    tokens_per_doc: int | str = "all"

def getattr_nested(obj, attr_path):
    for attr in attr_path.split('.'):
        obj = getattr(obj, attr)
    return obj

def vllm_select_tokens(llm, rep, input_ids, tokens_per_doc="all", exclude_bos=True, doc_idxs=None):
    bos_indices = (input_ids == llm.tokenizer.bos_token_id).nonzero(as_tuple=True)[0]
    starts = bos_indices.tolist()
    ends = starts[1:] + [rep.shape[0]]
    seqs = [rep[start:end] for start, end in zip(starts, ends)]
    # remove bos token
    if exclude_bos:
        seqs = [seq[1:] for seq in seqs]
    # note that short docs might return less than tokens_per_doc tokens
    if tokens_per_doc == "all":
        token_idxs = [list(range(len(seq))) for seq in seqs]
        rep_selected = seqs
    elif tokens_per_doc == "last":
        token_idxs = [[len(seq) - 1] for seq in seqs]
        rep_selected = [seq[idx] for seq, idx in zip(seqs, token_idxs)]
    else:
        token_idxs = [np.random.permutation(len(seq)).tolist() for seq in seqs]
        token_idxs = [idx[:tokens_per_doc] for idx in token_idxs]
        rep_selected = [seq[idx] for seq, idx in zip(seqs, token_idxs)]
    # collect and flatten doc_idxs
    if doc_idxs is not None:
        doc_idxs = sum([[doc_idxs[i]] * len(token_idxs[i]) for i in range((len(rep_selected)))], [])
    # flatten token_idxs
    token_idxs = sum(token_idxs, [])
    return torch.cat(rep_selected, dim=0), token_idxs, doc_idxs

@torch.no_grad()
def vllm_get_rep(llm, hook_kwargs, batch, tokens_per_doc="all", doc_idxs=None, hook_mode="output"):
    assert "Llama" in llm.model.config._name_or_path, "vllm separately returns hidden and residual for Llama models; check for others"
    batch = [{'prompt_token_ids': ids} for ids in batch]
    embed_layer = hook_kwargs["input"].get("embed_layer", "model.embed_tokens")
    with llm.trace(max_new_tokens=0, remote=False) as tracer:
        with tracer.invoke(batch):
            # NOTE: input_ids is the same as scheduled_ids; vllm doesn't shuffle anything
            scheduled_ids = getattr_nested(llm, embed_layer).input.detach().clone().save()
            input_ids = scheduled_ids
            # NOTE: list comprehensions have weird accrual behavior in nnsight
            # DO NOT do [x[0].detach().clone() + x[1].detach().clone() for x in rep]; you need the explicit for loop below
            rep = []
            for k in hook_kwargs["input"]["layers"]:
                if hook_mode == "output":
                    out = getattr_nested(llm, k).output
                    # https://github.com/vllm-project/vllm/blob/9ccbf6b692e0e39995b063a8381a322097cff5e0/vllm/model_executor/models/llama.py#L347
                    # vllm returns the hidden_state and residual separately
                    rep.append(out[0].detach().cpu().clone() + out[1].detach().cpu().clone())
                elif hook_mode == "input":
                    out = getattr_nested(llm, k).input
                    rep.append(out.detach().cpu().clone())
            # NOTE: multi-layer defaults to diff token for each layer
            rep_l, token_idxs_l, doc_idxs_l = [], [], []
            for x in rep:
                x, token_idxs_x, doc_idxs_x = vllm_select_tokens(
                    llm,
                    x,
                    input_ids,
                    tokens_per_doc=tokens_per_doc,
                    exclude_bos=True,
                    doc_idxs=doc_idxs,
                )
                rep_l.append(x)
                token_idxs_l.append(token_idxs_x)
                doc_idxs_l.append(doc_idxs_x)
            rep = rep_l.save()
            token_idxs = token_idxs_l.save()
            doc_idxs = doc_idxs_l.save()
            if rep[0].shape[0] != input_ids.shape[0] - len(batch):
                print(f"WARNING: rep and input_ids shape differ, rep={rep[0].shape[0]}, input_ids={input_ids.shape[0]}, bs={len(batch)}")
    return rep, token_idxs, doc_idxs

def get_train_dataloader(config, tokenizer):
    # load huggingface dataset
    train_dataset = load_dataset(
        path=config.dataset_path, 
        name=config.dataset_name, 
        split=config.dataset_split
    )
        
    # split docs into train and val
    np.random.seed(config.seed)
    random_idxs = np.random.permutation(range(len(train_dataset)))
    train_docs, val_docs = random_idxs[:-config.val_size].tolist(), random_idxs[-config.val_size:].tolist()
    json.dump(train_docs, open(f"{config.output_path}/train_docs.json", "w"))
    json.dump(val_docs, open(f"{config.output_path}/val_docs.json", "w"))

    train_dataset = train_dataset.select(train_docs)
    collate_fn = lambda rows: tokenizer(
        [row['text'] for row in rows],
        padding=False, 
        truncation=True, 
        max_length=config.max_length,
        # return_tensors='pt',
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        num_workers=0,
    )
    return train_dataloader, train_docs

def save_activations(config, device="cuda:0"):
    # setup output path
    if os.path.exists(config.output_path):
        print(f"Warning: {config.output_path} already exists. Deleting and overwriting it...")
        shutil.rmtree(config.output_path, ignore_errors=True)
    os.makedirs(config.output_path, exist_ok=True)

    # init wandb
    if config.wandb_enabled:
        print("Initializing wandb")
        import wandb
        wandb_run = wandb.init(
            entity=config.wandb_entity,
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=OmegaConf.to_container(config),
        )
    
    # load model
    max_num_batched_tokens = config.batch_size * config.max_length
    model = VLLM(
        config.model_name, 
        dispatch=True, 
        max_num_batched_tokens=max_num_batched_tokens,
        **config.get("vllm_kwargs", {})
    )
    model.to(device)
    tokenizer = model.tokenizer
    # NOTE: vllm default truncation side is left (which is very odd)
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    # setup dynamic act writers
    get_layer = lambda text: int(re.search(r'layers\.(\d+)', text).group(1))
    act_writers = [
        DynamicActWriter(
          f"{config.output_path}/layer_{get_layer(l):02d}",
          acts_per_shard=config.acts_per_shard,
          max_total_acts=config.max_total_acts,
        ) for l in config.latent_model_config["hook_kwargs"]["input"]["layers"]
    ]

    # setup dataloader
    train_dataloader, train_docs = get_train_dataloader(config, tokenizer)

    total_iters = len(train_dataloader)
    pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), miniters=100, smoothing=0)
    for it, batch in pbar:
        try:
            input_ids = batch["input_ids"]

            # remove empty docs
            doc_idxs = train_docs[it * config.batch_size : (it + 1) * config.batch_size]
            doc_idxs = [doc_idxs[i] for i in range(len(doc_idxs)) if len(input_ids[i]) > 1]
            input_ids = [input_ids[i] for i in range(len(input_ids)) if len(input_ids[i]) > 1]
            assert len(input_ids) == len(doc_idxs)
            if not doc_idxs:
                continue

            # run forward pass
            acts, token_idxs, doc_idxs = vllm_get_rep(
                model, 
                config.latent_model_config["hook_kwargs"], 
                input_ids, 
                tokens_per_doc=config.tokens_per_doc, 
                doc_idxs=doc_idxs, 
                hook_mode=config.latent_model_config["hook_mode"]
            )

            # write the act
            for l in range(len(acts)):
                act_writers[l].write(acts[l], token_idxs=token_idxs[l], doc_idxs=doc_idxs[l])
            
            # helper for multi-layer case to concat rep_statistics into a single file
            if len(act_writers) > 1:
                if not os.path.exists(f"{config.output_path}/rep_statistics.pt"):
                    paths = sorted([f"{str(act_writer.data_dir)}/rep_statistics.pt" for act_writer in act_writers])
                    if all([os.path.exists(path) for path in paths]):
                        stats = [torch.load(path) for path in paths]
                        torch.save(
                          {k: torch.stack([stat[k] for stat in stats]) for k in stats[0].keys()},
                          f"{config.output_path}/rep_statistics.pt"
                        )

            # wandb log
            if config.wandb_enabled and it % 10 == 0:
                process = psutil.Process(os.getpid())
                cpu_mem_mb = process.memory_info().rss / 1024 / 1024
                gpu_mem_mb = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
                gpu_mem_reserved_mb = torch.cuda.memory_reserved() / 1024 / 1024 if torch.cuda.is_available() else 0
                wandb_run.log({
                    "memory/cpu_mb": cpu_mem_mb,
                    "memory/gpu_allocated_mb": gpu_mem_mb,
                    "memory/gpu_reserved_mb": gpu_mem_reserved_mb,
                    "iteration": it,
                    "total_iterations": total_iters,
                })
        except Exception as e:
            print(f"Error at iteration {it}: {e}")
            continue
    for act_writer in act_writers:
        act_writer.flush()

if __name__ == "__main__":
    config_base = OmegaConf.structured(SaveActivationsConfig)
    OmegaConf.set_struct(config_base, False)
    config_cli = OmegaConf.from_cli()
    config_path = config_cli.pop("config", None)
    config_file = OmegaConf.load(config_path) if config_path else OmegaConf.create()
    config = OmegaConf.merge(config_base, config_file, config_cli)
    OmegaConf.resolve(config)
    save_activations(config)
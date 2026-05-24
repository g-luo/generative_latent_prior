from dataclasses import dataclass, field
from datasets import load_dataset
import fcntl
import json
import numpy as np
from omegaconf import OmegaConf
import os
from pathlib import Path
import psutil
import re
import shutil
import time
from tqdm import tqdm
from typing import Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, IterableDataset
from nnsight.modeling.vllm import VLLM

from glp_train import ActDataset
from glp.utils_acts import MemmapReader, MemmapWriter

@dataclass
class SaveActivationsConfig:
    """
    Configuration for saving activations.
    """
    # model
    output_path: str
    model_name: str
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

def getattr_nested(obj, attr_path):
    for attr in attr_path.split('.'):
        obj = getattr(obj, attr)
    return obj

def vllm_unpack(llm, rep, input_ids, padding_value=0):
    bos_indices = (input_ids == llm.tokenizer.bos_token_id).nonzero(as_tuple=True)[0]
    starts = bos_indices.tolist()
    ends = starts[1:] + [rep.shape[0]]
    seqs = [rep[start:end] for start, end in zip(starts, ends)]
    # pad to longest sequence
    # NOTE: you should never use the pad acts for anything
    # it won't be the same as pad_id -> pad_acts
    padding_side = getattr(llm.tokenizer, "padding_side")
    if padding_side == "left":
        rep_padded = pad_sequence([s.flip(0) for s in seqs], batch_first=True, padding_value=padding_value).flip(1)
    elif padding_side == "right":
        rep_padded = pad_sequence(seqs, batch_first=True, padding_value=padding_value)
    return rep_padded

def vllm_select_n(llm, rep, input_ids, n=None, exclude_bos=True, doc_idxs=None):
    bos_indices = (input_ids == llm.tokenizer.bos_token_id).nonzero(as_tuple=True)[0]
    starts = bos_indices.tolist()
    ends = starts[1:] + [rep.shape[0]]
    seqs = [rep[start:end] for start, end in zip(starts, ends)]
    # remove bos token
    if exclude_bos:
        seqs = [seq[1:] for seq in seqs]
    # note that short docs might return less than n tokens
    if n == "last":
        token_idxs = [[len(seq) - 1] for seq in seqs]
        rep_selected = [seq[idx] for seq, idx in zip(seqs, token_idxs)]
    elif n is not None:
        token_idxs = [np.random.permutation(len(seq)).tolist() for seq in seqs]
        token_idxs = [idx[:n] for idx in token_idxs]
        rep_selected = [seq[idx] for seq, idx in zip(seqs, token_idxs)]
    else:
        token_idxs = [list(range(len(seq))) for seq in seqs]
        rep_selected = seqs
    # collect and flatten doc_idxs
    if doc_idxs is not None:
        doc_idxs = sum([[doc_idxs[i]] * len(token_idxs[i]) for i in range((len(rep_selected)))], [])
    # flatten token_idxs
    token_idxs = sum(token_idxs, [])
    return torch.cat(rep_selected, dim=0), token_idxs, doc_idxs

@torch.no_grad()
def vllm_get_rep(llm, hook_kwargs, batch, tokens_per_doc=None, doc_idxs=None, hook_mode="output"):
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
                x, token_idxs_x, doc_idxs_x = vllm_select_n(llm, x, input_ids, tokens_per_doc, exclude_bos=True, doc_idxs=doc_idxs)
                rep_l.append(x)
                token_idxs_l.append(token_idxs_x)
                doc_idxs_l.append(doc_idxs_x)
            rep = rep_l.save()
            token_idxs = token_idxs_l.save()
            doc_idxs = doc_idxs_l.save()
            if rep[0].shape[0] != input_ids.shape[0] - len(batch):
                print(f"WARNING: rep and input_ids shape differ, rep={rep[0].shape[0]}, input_ids={input_ids.shape[0]}, bs={len(batch)}")
    return rep, token_idxs, doc_idxs

class DynamicActDataset(IterableDataset):
    def __init__(self, data_dir, embedding_dim, dtype=np.int16):
        super().__init__()
        self.data_dir = data_dir
        self.seen_shards = set()
        self.dtype = dtype
        self.embedding_dim = embedding_dim

    def __iter__(self):
        """
        The producer generates endlessly in a loop;
        the consumer just needs to mark the shard to (1) tell producer "do not delete" and (2) tell other consumers to party with it.
        The consumer also shouldn't revisit shards.
        """
        last_error = None
        while True:
            try:
                # prioritize smallest active then smallest ready
                # active shards are trying to start a party and get all the consumers to go there
                active_shards = sorted(list(self.data_dir.glob("*.active")))
                ready_shards = sorted(list(self.data_dir.glob("*.ready")))
                candidates = [f for f in (active_shards + ready_shards) if f.name not in self.seen_shards]
                if not candidates:
                    curr_error = f"Consumer {self.data_dir}: no candidates found, saw: {active_shards + ready_shards} with seen: {self.seen_shards}"
                    if curr_error != last_error:
                        print(curr_error)
                        last_error = curr_error
                    time.sleep(1)
                    continue
                best_shard = candidates[0]
                batch_id = best_shard.stem
                current_flag = best_shard
                # if is ready, promote to active
                if best_shard.suffix == ".ready":
                    active_path = self.data_dir / f"{batch_id}.active"
                    try:
                        os.rename(best_shard, active_path)
                        current_flag = active_path
                    except (FileNotFoundError, PermissionError):
                        # someone else moved/deleted it, loop back to re-scan
                        continue
                f = open(current_flag / ".lock", 'r')
                try:
                    # LOCK_SH ensures we share the file with other consumers
                    fcntl.flock(f, fcntl.LOCK_SH)
                    reader = MemmapReader(self.data_dir / f"{batch_id}.active", dtype=self.dtype)
                    act_dataset = ActDataset(
                        reader=reader, 
                        embedding_dim=self.embedding_dim
                    )
                    # TODO: seed based on shard
                    # np.random.seed(int(batch_id.split("_")[-1]))
                    # randomly shuffle indices to avoid tokens from same doc
                    for idx in np.random.permutation(range(len(act_dataset))):
                        yield act_dataset.__getitem__(idx)
                    # close reader and avoid observing same shard again
                    self.seen_shards.add(current_flag.name)
                except Exception as e:
                    curr_error = str(e)
                    if curr_error != last_error:
                        print(f"Consumer {self.data_dir}: {e}")
                        last_error = curr_error
                    time.sleep(1)
                    continue
                finally:
                    f.close()
            except Exception as e:
                curr_error = str(e)
                if curr_error != last_error:
                    print(f"Consumer {self.data_dir}: {e}")
                    last_error = curr_error
                time.sleep(1)
                continue

class MixedDynamicActDataset(IterableDataset):
    def __init__(self, data_dirs, embedding_dim, dtype=np.int16):
        super().__init__()
        self.datasets = [DynamicActDataset(data_dir, embedding_dim, dtype) for data_dir in data_dirs]
        self.embedding_dim = embedding_dim
        self.dtype = dtype

    def __iter__(self):
        last_error = None
        iterators = [iter(ds) for ds in self.datasets]
        while True:
            random_idxs = np.random.permutation(range(len(iterators)))
            for i in random_idxs:
                try:
                    yield next(iterators[i])
                except Exception as e:
                    print(f"Consumer {self.datasets[i].data_dir}: {e}")
                    continue

class DynamicActWriter:
    def __init__(self, data_dir, acts_per_shard, max_total_acts, dtype=np.int16):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.acts_per_shard = acts_per_shard
        self.max_shards = max_total_acts // acts_per_shard
        
        self.current_writer = None
        self.current_shard_id = None

        self.token_buffer = []
        self.doc_buffer = []

        # NOTE: ideally dtype is saved in a meta file and automatically known
        # but for now we just assume we use only with DynamicActDataset
        self.dtype = dtype

        self.mo1 = None
        self.mo2 = None
        self.c_count = 0

    def _rotate_shard(self):
        # commit current shard once full
        if self.current_writer:
            self.flush()
            # convert from tmp to ready
            temp_dir = self.data_dir / f"{self.current_shard_id}.tmp"
            final_dir = self.data_dir / f"{self.current_shard_id}.ready"
            os.rename(temp_dir, final_dir)
            # init global mean / var if doesn't exist
            if not os.path.exists(self.data_dir / "rep_statistics.pt"):
                shutil.copy(final_dir / "rep_statistics.pt", self.data_dir / "rep_statistics.pt")
            print(f"\nProducer: committed {self.current_shard_id}")

        # remove oldest shard if full
        while True:
            shards = sorted(list(self.data_dir.glob("*.ready")) + list(self.data_dir.glob("*.active")))
            if len(shards) < self.max_shards:
                break
            # only do ejection if there are active shards
            if len(list(self.data_dir.glob("*.active"))) == 0:
                print(f"Producer: need some file to be active before starting ejection")
                time.sleep(5)
                continue
            oldest_flag = shards[0]
            f = open(oldest_flag / ".lock", 'r')
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                print(f"Producer: ejecting tail shard: {oldest_flag.name}")
                shutil.rmtree(oldest_flag, ignore_errors=True)
                break
            except (BlockingIOError, IOError):
                # edge case: oldest shard is active
                # producer is blocked until the consumer moves to the next shard
                print(f"Producer: tail shard {oldest_flag.name} is ACTIVE. Waiting...", end="")
                time.sleep(5)
                continue

        # start new shard
        self.current_shard_id = f"shard_{int(time.time() * 1000):015d}"
        temp_dir = self.data_dir / f"{self.current_shard_id}.tmp"
        temp_dir.mkdir()
        lock_file = temp_dir / ".lock"
        lock_file.touch()
        self.current_writer = MemmapWriter(output_dir=temp_dir, dtype=self.dtype, file_size=2**30)
        self.token_buffer = []
        self.doc_buffer = []
        # reset stats
        self.mo1 = None
        self.mo2 = None
        self.c_count = 0

    def write(self, acts, token_idxs=[], doc_idxs=[]):
        n_count = acts.shape[0]
        if self.current_writer is None or (self.c_count + n_count) > self.acts_per_shard:
            self._rotate_shard()

        self.token_buffer.extend(token_idxs)
        self.doc_buffer.extend(doc_idxs)
        for i in range(n_count):
            activation = acts[i, :]
            # handle dtypes since vllm computes in bf16
            if self.dtype == np.int16:
                activation = activation.view(torch.int16)
            elif self.dtype == np.float32:
                activation = activation.to(torch.float32)
            else:
                raise NotImplementedError
            activation = activation.cpu().numpy()
            # actually write to disk
            self.current_writer.write(activation)
    
        self.update_stats(acts)
        self.c_count += n_count

    def update_stats(self, acts):
        # compute mean and var in fp32
        acts = acts.to(torch.float32)
        if self.mo1 is None:
            self.mo1 = torch.zeros_like(acts[0, :])
            self.mo2 = torch.zeros_like(acts[0, :])
        n_mo1 = acts[:, :].mean(dim=0)
        n_mo2 = (acts[:, :]**2).mean(dim=0)
        n_count = acts.shape[0]
        c_count = self.c_count
        self.mo1 = self.mo1 * (c_count / (c_count + n_count)) + n_mo1 * (n_count / (c_count + n_count))
        self.mo2 = self.mo2 * (c_count / (c_count + n_count)) + n_mo2 * (n_count / (c_count + n_count))
        
    def flush(self):
        self.current_writer.flush()
        temp_dir = self.data_dir / f"{self.current_shard_id}.tmp"
        json.dump(self.token_buffer, open(temp_dir / "token_buffer.json", "w"))
        json.dump(self.doc_buffer, open(temp_dir / "doc_buffer.json", "w"))
        self.mo2 -= self.mo1 ** 2
        torch.save({
            "mean": self.mo1.cpu(),
            "var": self.mo2.cpu(),
        }, temp_dir / "rep_statistics.pt")

# TODO: save and load dtype
# def load_dynamic_dataset(
#     dataset_paths: str | list[str],
#     embedding_dim: int
# ):
#     from save_buffer import MixedDynamicActDataset
#     dataset_paths = [dataset_paths] if isinstance(dataset_paths, str) else dataset_paths
#     dataset_paths = [get_path(path) for path in dataset_paths]
#     return MixedDynamicActDataset(data_dirs=dataset_paths, embedding_dim=embedding_dim)
# # Block if rep_statistic doesn't exist in dynamic case
# if "rep_statistic" in config and type(config.rep_statistic) is str and not os.path.exists(config.rep_statistic):
#     print(f"Waiting for rep_statistic {config.rep_statistic} to be generated...")
#     while not os.path.exists(config.rep_statistic):
#         time.sleep(5)

def get_train_dataloader(config, tokenizer):
    # load huggingface dataset
    train_dataset = load_dataset(
      dataset_path=config.dataset_path, 
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
    if os.path.exists(config.config.output_path):
        print(f"Warning: {config.config.output_path} already exists. Deleting and overwriting it...")
        shutil.rmtree(config.config.output_path, ignore_errors=True)
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

# TODO: check gpu size
# TODO: getattr nested
# TODO: race condition
"""Activation datasets, including the dynamic producer-consumer pipeline.

How the dynamic pipeline works:

- Producer (`glp_save.py`): runs the LLM over FineWeb documents, caches the
  hooked-layer activations, and writes them into a sliding shard buffer under
  `data/<name>/layer_XX/` (`DynamicActWriter`). Shards are written as `.tmp`,
  committed as `.ready` (`acts_per_shard` activations each), and the buffer is
  capped at `max_total_acts`: once full, the producer ejects the oldest
  non-active shard before starting a new one. It also maintains running
  `rep_statistics.pt` (mean/std) used for activation normalization.

- Trainer (`glp_train.py`): first waits for `rep_statistics.pt` to appear,
  then streams shards from the buffer (`DynamicActDataset`), promoting
  `.ready` shards to `.active` (which lock-protects them from ejection) and
  never revisiting a shard. This means activations are seen at most once. If the
  trainer outpaces the producer it idles until a new shard is committed; if
  the producer outpaces the trainer it blocks on ejecting the active shard.

- Orchestration: launch both processes per the README (producer on one GPU,
  trainer on another, e.g. from a `run_full.sh`-style script or two tmux
  panes). The trainer can be started before the producer has written
  anything; it will wait. Since the dynamic stream is infinite, `epoch_size`
  (1M activations) defines an "epoch" and `num_epochs`/`save_epochs` bound
  the run (defaults: 1024 epochs ~= 1B activations, log-scale checkpoints).
"""
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import time

import numpy as np
from omegaconf import ListConfig
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, IterableDataset

from glp.denoiser import Normalizer
from glp.utils_acts import MemmapReader, MemmapWriter

def read_dtype(path):
    dtype_path = Path(path) / "dtype.txt"
    return np.dtype(dtype_path.read_text().strip().replace('np.', ''))

def write_dtype(path, dtype):
    dtype_path = Path(path) / "dtype.txt"
    dtype_path.write_text(str(np.dtype(dtype)))

# =======================
#    Static Dataset
# =======================
class ActDataset(Dataset):
    def __init__(self, reader: MemmapReader | list[MemmapReader]):
        reader = [reader] if not isinstance(reader, (list, ListConfig)) else reader
        self.reader = reader

    def __len__(self):
        return len(self.reader[0])

    def __getitem__(self, idx):
        batch = {}
        # handle multi_layer model
        # folders should be of the form layer_<idx>
        # also need to set multi_layer_n_layers in glp_kwargs
        # for this to actually be used by denoiser
        layer_match = re.search(r"layer_(\d+)", str(self.reader[0].data_dir))
        if layer_match:
            batch["layer_idx"] = int(layer_match.group(1))
        # prepare latents
        # latents should be saved as (dim,)
        latents = [
            torch.tensor(reader[idx])[None, :]
            for r, reader in enumerate(self.reader)
        ]
        # handle special multi-reader case
        # e.g., concat different features from different readers
        # not currently used but useful for conditional modeling
        latents = torch.cat(latents, dim=-1)
        # handle data saved in half rather than full precision
        latents = latents.view(torch.bfloat16) if latents.dtype == torch.int16 else latents
        latents = latents.float()
        batch["activations"] = latents
        return batch

class ActivationCollator:
    def __init__(self, normalizer: Normalizer):
        self.normalizer = normalizer

        # check if the normalizer has per-layer stats (multi-layer model)
        self.is_multi_layer = normalizer.mean.ndim > 1 and normalizer.mean.shape[0] > 1

    @torch.no_grad()
    def __call__(self, rows):
        batch = {}
        # handle multi_layer model
        if 'layer_idx' in rows[0] and self.is_multi_layer:
            layer_idx = torch.tensor([row['layer_idx'] for row in rows], dtype=torch.long)
            batch['layer_idx'] = layer_idx
        else:
            layer_idx = None
        # prepare latents
        latents = torch.stack([row['activations'] for row in rows], dim=0)
        batch['latents'] = self.normalizer.normalize(latents, layer_idx=layer_idx)
        return batch

# =======================
#    Dynamic Dataset
# =======================
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

        self.dtype = dtype
        write_dtype(self.data_dir, self.dtype)

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
            try:
                f = open(oldest_flag / ".lock", 'r')
            except FileNotFoundError:
                # a consumer promoted (.ready -> .active) or ejected this shard
                # between the glob above and here; re-scan instead of crashing
                continue
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
            finally:
                f.close()

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
        # nothing was ever written (e.g., every batch errored out)
        if self.current_writer is None:
            return
        self.current_writer.flush()
        temp_dir = self.data_dir / f"{self.current_shard_id}.tmp"
        json.dump(self.token_buffer, open(temp_dir / "token_buffer.json", "w"))
        json.dump(self.doc_buffer, open(temp_dir / "doc_buffer.json", "w"))
        self.mo2 -= self.mo1 ** 2
        torch.save({
            "mean": self.mo1.cpu(),
            "var": self.mo2.cpu(),
        }, temp_dir / "rep_statistics.pt")

class DynamicActDataset(IterableDataset):
    def __init__(self, data_dir, dtype=np.int16):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.seen_shards = set()
        self.dtype = dtype

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
                    act_dataset = ActDataset(reader=reader)
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
    def __init__(self, data_dirs):
        super().__init__()
        self.datasets = [
            DynamicActDataset(data_dir, read_dtype(data_dir))
            for data_dir in data_dirs
        ]

    def __iter__(self):
        iterators = [iter(ds) for ds in self.datasets]
        while True:
            random_idxs = np.random.permutation(range(len(iterators)))
            for i in random_idxs:
                try:
                    yield next(iterators[i])
                except Exception as e:
                    print(f"Consumer {self.datasets[i].data_dir}: {e}")
                    continue

# =========================
#     Shared Dataloader
# =========================
def load_activation_dataset(
    dataset_paths: str | list[str],
    dynamic: bool = False,
):
    dataset_paths = [dataset_paths] if isinstance(dataset_paths, str) else dataset_paths
    dataset_paths = [Path(path) for path in dataset_paths]
    # dynamic case
    if dynamic:
        return MixedDynamicActDataset(dataset_paths)
    # static case
    datasets = []
    for path in dataset_paths:
        dtype = read_dtype(path)
        reader = MemmapReader(path, dtype)
        dataset = ActDataset(reader=reader)
        datasets.append(dataset)
    return ConcatDataset(datasets)

def get_activation_dataloader(
    dataset,
    batch_size: int,
    normalizer: Normalizer,
    shuffle: bool = True,
):
    # dynamic dataset already natively shuffled
    if isinstance(dataset, IterableDataset):
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        collate_fn=ActivationCollator(normalizer),
        num_workers=0,
        pin_memory=False,
    )

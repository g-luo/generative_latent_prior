import logging
import math
import os
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, List, Optional

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from glp.denoiser import GLP
from glp_dataset import (
    get_activation_dataloader,
    load_activation_dataset,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainConfig:
    # model
    model_name: str = ""
    glp_kwargs: Optional[Any] = None
    # data
    shuffle: bool = True
    train_dataset: Any = ""  # str or list[str] (multi-layer)
    rep_statistic: str = ""
    dynamic_dataset: bool = False  # stream from the producer-consumer shard buffer
    # training
    use_bf16: bool = True
    num_epochs: int = 1
    epoch_size: Optional[int] = None
    batch_size: int = 4096
    learning_rate: float = 5e-5
    lr_scheduler: Optional[dict] = None
    gradient_accumulation_steps: int = 1
    gradient_clipping_threshold: float = 1
    # logging and saving
    log_every_n_steps: int = 10
    save_every_n_steps: Optional[int] = None
    save_epochs: Optional[List[int]] = None
    save_opt_state: bool = False
    output_path: Optional[Path] = None
    # wandb
    wandb_enabled: bool = False
    wandb_entity: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None

def linear_scheduler(step, max_steps, initial_factor, final_factor):
    alpha = step / max_steps
    return alpha * final_factor + (1 - alpha) * initial_factor

def linear_scheduler_with_warmup(step, *, warmup_steps, max_steps, initial_factor, final_factor):
    if step < warmup_steps:
        return linear_scheduler(step, warmup_steps, initial_factor, 1.0)
    elif step >= max_steps:
        return final_factor
    else:
        return linear_scheduler(step - warmup_steps, max_steps - warmup_steps, 1.0, final_factor)

def cosine_scheduler(step, max_steps, initial_factor, final_factor):
    alpha = step / max_steps
    cosine_out = 0.5 * (1 + math.cos(math.pi * alpha))
    return final_factor + (initial_factor - final_factor) * cosine_out

def cosine_scheduler_with_warmup(step, *, warmup_steps, max_steps, initial_factor, final_factor):
    if step < warmup_steps:
        return linear_scheduler(step, warmup_steps, initial_factor, 1.0)
    elif step >= max_steps:
        return final_factor
    else:
        return cosine_scheduler(step - warmup_steps, max_steps - warmup_steps, 1.0, final_factor)

def main(device="cuda:0"):
    config_base = OmegaConf.structured(TrainConfig())
    OmegaConf.set_struct(config_base, False)
    config_cli = OmegaConf.from_cli()
    config_path = config_cli.pop("config", None)
    config_file = OmegaConf.load(config_path) if config_path else OmegaConf.create()
    config = OmegaConf.merge(config_base, config_file, config_cli)

    # setup output path
    output_path = Path(config.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving checkpoints to {output_path}")
    # resolve=True bakes in interpolations (incl. ${oc.env:...}) for provenance
    OmegaConf.save(config, output_path / "config.yaml", resolve=True)

    # wait for rep_statistic from producer
    rep_statistic = config.glp_kwargs.get("normalizer_config", {}).get("rep_statistic")
    if rep_statistic and not os.path.exists(rep_statistic):
        logger.info(f"Waiting for rep_statistic {rep_statistic}...")
        while not os.path.exists(rep_statistic):
            time.sleep(5)

    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    logger.info(f"Config: {config}")

    # init wandb
    wandb_run = None
    if config.wandb_enabled:
        import wandb
        wandb_run = wandb.init(
            entity=config.wandb_entity,
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=OmegaConf.to_container(config),
        )

    # load model
    model = GLP(**config.glp_kwargs)
    model.to(device)
    logger.info(f"Model param count: {sum(p.numel() for p in model.parameters())}")

    # load dataset
    train_dataset = load_activation_dataset(
        config.train_dataset,
        dynamic=config.dynamic_dataset
    )
    train_dataloader = get_activation_dataloader(
        dataset=train_dataset,
        batch_size=config.batch_size // config.gradient_accumulation_steps,
        normalizer=model.normalizer,
        shuffle=config.shuffle,
    )

    # setup optimizer and scheduler
    epoch_size = (config.epoch_size // config.batch_size) if config.epoch_size else len(train_dataloader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    if config.lr_scheduler is None:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1)
    else:
        total_num_steps = config.num_epochs * (epoch_size // config.gradient_accumulation_steps)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=partial(
                eval(config.lr_scheduler["scheduler_cls"]),
                warmup_steps=config.lr_scheduler["warmup_ratio"] * total_num_steps,
                max_steps=total_num_steps,
                initial_factor=config.lr_scheduler["initial_factor"],
                final_factor=config.lr_scheduler["final_factor"],
            )
        )

    # training loop
    train_steps = 0
    num_gradient_steps = 0
    
    for epoch in range(config.num_epochs):
        model.train()
        gradient_steps_in_epoch = epoch_size // config.gradient_accumulation_steps
        pbar = tqdm(
            total=gradient_steps_in_epoch,
            desc=f"Training Epoch: {epoch + 1}",
            dynamic_ncols=True,
        )
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) if v is not None else None for k, v in batch.items()}

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=config.use_bf16):
                outputs = model(**batch)
                loss = outputs.loss

            loss = loss / config.gradient_accumulation_steps
            loss.backward()
            train_steps += 1

            if train_steps % config.gradient_accumulation_steps == 0:
                num_gradient_steps += 1

                if config.gradient_clipping_threshold > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.gradient_clipping_threshold
                    )

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                pbar.update(1)
                pbar.set_description(
                    f"Epoch: {epoch + 1}/{config.num_epochs}, "
                    f"batch {step + 1}/{epoch_size} "
                    f"(loss: {loss.detach().float():.4f})"
                )

                if num_gradient_steps % config.log_every_n_steps == 0:
                    avg_loss = loss.detach().item()
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "train/epoch": epoch,
                                "train/step": num_gradient_steps,
                                "train/loss": avg_loss,
                                "train/learning_rate": scheduler.get_last_lr()[0],
                            },
                            step=num_gradient_steps
                        )

                if config.save_every_n_steps and num_gradient_steps % config.save_every_n_steps == 0:
                    save_checkpoint(model, output_path, f"step_{num_gradient_steps}", optimizer, scheduler, config)

            if step >= gradient_steps_in_epoch * config.gradient_accumulation_steps:
                break

        pbar.close()

        # save epoch checkpoint
        if config.save_epochs and (epoch + 1) in set(config.save_epochs):
            save_checkpoint(model, output_path / "checkpoints", f"epoch_{epoch + 1}")
        
        # always save latest checkpoint
        save_checkpoint(model, output_path, "final", optimizer, scheduler, save_opt_state=config.save_opt_state)

    if wandb_run is not None:
        wandb.finish()

def save_checkpoint(model, output_path, checkpoint_name, optimizer=None, scheduler=None, save_opt_state=False):
    model.save_pretrained(path=output_path, name=checkpoint_name)
    logger.info(f"Model saved to {output_path}/{checkpoint_name}")
    if save_opt_state:
        if optimizer is not None:
            torch.save(optimizer.state_dict(), output_path / "optimizer_state.pt")
        if scheduler is not None:
            torch.save(scheduler.state_dict(), output_path / "scheduler_state.pt")

if __name__ == "__main__":
    main()
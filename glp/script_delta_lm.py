from dataclasses import dataclass, field
import json
from omegaconf import OmegaConf
import numpy as np
import os
import torch
import torch.nn.functional as F
import warnings

from baukit import TraceDict
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from glp.denoiser import load_glp
from glp.script_steer import postprocess_on_manifold_wrapper

# =======================
#     Interventions
# =======================
def glp_intervention_wrapper(postprocess_fn):
    """
    Replace the activations with their GLP noise-and-denoise reconstruction.
    See `postprocess_on_manifold_wrapper` in script_steer.py.
    """
    def intervention(output, layer_name, inputs):
        act = output[0] if isinstance(output, tuple) else output
        # exclude bos
        act[:, 1:] = postprocess_fn(act[:, 1:])
        return (act, *output[1:]) if isinstance(output, tuple) else act
    return intervention

def sae_intervention_wrapper(sae_model):
    """
    Replace the activations with their SAE reconstruction.
    """
    def intervention(output, layer_name, inputs):
        act = output[0] if isinstance(output, tuple) else output
        # exclude bos
        act[:, 1:] = sae_model(act[:, 1:].to(sae_model.dtype)).to(act.dtype)
        return (act, *output[1:]) if isinstance(output, tuple) else act
    return intervention

def fold_llama_scope_norm(sae_model, release, sae_id):
    """
    Llama Scope SAEs expect activations rescaled to mean norm sqrt(d_in). sae_lens folds this factor into the
    weights in SAE.from_pretrained, except in versions 6.34.2-6.37.0 (our pin) where that step is missing.
    This is the block that sae_lens 6.38.0 restored, reading the factor from its own pretrained_saes.yaml.
    Reference: https://github.com/decoderesearch/SAELens/blob/v6.38.0/sae_lens/saes/sae.py (from_pretrained_with_cfg_and_sparsity)
    """
    if sae_model.cfg.normalize_activations == "expected_average_only_in":
        import importlib.resources, yaml
        directory = yaml.safe_load(importlib.resources.files("sae_lens").joinpath("pretrained_saes.yaml").read_text())
        norm_scaling_factor = next((sae.get("norm_scaling_factor") for sae in directory[release]["saes"] if sae["id"] == sae_id), None)
        if norm_scaling_factor is not None:
            sae_model.fold_activation_norm_scaling_factor(norm_scaling_factor)
            sae_model.cfg.normalize_activations = "none"
        else:
            warnings.warn(f"norm_scaling_factor not found for {release} and {sae_id}, but normalize_activations is 'expected_average_only_in'. Skipping normalization folding.")

# =======================
#        LM Loss
# =======================
@torch.no_grad()
def lm_loss(hf_model, hf_tokenizer, texts, layer, intervention_fn, batch_size, max_length, device):
    losses = []
    for i in range(0, len(texts), batch_size):
        inputs = hf_tokenizer(texts[i:i + batch_size], return_tensors="pt", padding="max_length", truncation=True, max_length=max_length).to(device)
        # ignore padding in the loss
        labels = inputs["input_ids"].clone()
        labels[inputs["attention_mask"] == 0] = -100
        with TraceDict(hf_model, layers=[f"model.layers.{layer}"], edit_output=intervention_fn):
            logits = hf_model(**inputs).logits
        # per-doc mean next-token loss
        loss = F.cross_entropy(logits[:, :-1].transpose(1, 2), labels[:, 1:], reduction="none")
        losses += loss.mean(dim=1).tolist()
    return losses

def load_openwebtext_docs(config):
    dataset = load_dataset(config.dataset_path, split="train")
    np.random.seed(config.docs_seed)
    random_idxs = np.random.permutation(range(len(dataset)))
    heldout_docs = random_idxs[config.docs_offset:config.docs_offset + config.num_docs]
    return dataset.select(heldout_docs)["text"]

@dataclass
class DeltaLMConfig:
    save_folder: str = "runs/delta_lm"
    model_name: str = "meta-llama/Llama-3.1-8B" # or meta-llama/Llama-3.1-8B-Instruct for the transfer setting
    weights_folder: str | None = "generative-latent-prior/glp-llama8b-d6"
    ckpt_name: str | None = "final"
    sae_release: str = "llama_scope_lxr_32x"
    sae_id: str = "l15r_32x"
    dataset_path: str = "Skylion007/openwebtext"
    docs_seed: int = 0
    docs_offset: int = 2048 # held-out docs are a seed-0 permutation of the train split, starting after the first 2048
    num_docs: int = 2048 # set this to a small number for a quick test
    modes: list[str] = field(default_factory=lambda: ["base", "glp", "sae"])
    max_length: int = 128
    batch_size: int = 10
    u: float = 0.5
    num_timesteps: int = 20
    dtype: str = "float32"
    seed: int = 42

def evaluate_delta_lm(device="cuda:0"):
    default_config = OmegaConf.structured(DeltaLMConfig)
    OmegaConf.set_struct(default_config, False)
    config = OmegaConf.merge(default_config, OmegaConf.from_cli())
    torch.manual_seed(config.seed)

    texts = load_openwebtext_docs(config)

    # intervene on the layer the glp was trained on
    model = load_glp(config.weights_folder, device=device, checkpoint=config.ckpt_name)
    layer = dict(model.tracedict_config)["layers"][0]

    hf_tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    hf_tokenizer.pad_token = hf_tokenizer.eos_token
    hf_tokenizer.padding_side = "right"
    hf_tokenizer.truncation_side = "right"
    hf_model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=getattr(torch, config.dtype)).to(device).eval()

    interventions = {}
    for mode in config.modes:
        if mode == "base":
            interventions[mode] = None
        elif mode == "glp":
            interventions[mode] = glp_intervention_wrapper(postprocess_on_manifold_wrapper(model, u=config.u, num_timesteps=config.num_timesteps))
        elif mode == "sae":
            from sae_lens import SAE
            sae_model = SAE.from_pretrained(release=config.sae_release, sae_id=config.sae_id)
            sae_model = (sae_model[0] if isinstance(sae_model, tuple) else sae_model).to(device)
            fold_llama_scope_norm(sae_model, config.sae_release, config.sae_id)
            interventions[mode] = sae_intervention_wrapper(sae_model)
        else:
            raise NotImplementedError

    # saved like script_eval.py: {save_folder}/{model_name}/{weights_name}/{ckpt_name}.json holds the result,
    # with the per-doc losses of each arm alongside so an interrupted run resumes per arm
    weights_name = os.path.basename(config.weights_folder)
    save_dir = f"{config.save_folder}/{os.path.basename(config.model_name)}/{weights_name}"
    os.makedirs(save_dir, exist_ok=True)
    results = {}
    for mode, intervention_fn in interventions.items():
        save_file = f"{save_dir}/{config.ckpt_name}_{mode}.json"
        if os.path.exists(save_file):
            results[mode] = json.load(open(save_file))
            continue
        results[mode] = lm_loss(hf_model, hf_tokenizer, texts, layer, intervention_fn, config.batch_size, config.max_length, device)
        json.dump(results[mode], open(save_file, "w"))
    OmegaConf.save(config, f"{save_dir}/config.yaml")

    # delta lm loss = mean loss with intervention - mean loss without
    base_loss = float(np.mean(results["base"]))
    summary = {"base_loss": base_loss}
    for mode in config.modes:
        if mode != "base":
            summary[f"delta_{mode}"] = float(np.mean(results[mode]) - base_loss)
    json.dump(summary, open(f"{save_dir}/{config.ckpt_name}.json", "w"))
    print(summary)

if __name__ == "__main__":
    evaluate_delta_lm()

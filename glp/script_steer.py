from baukit import TraceDict
import einops
import pandas as pd
import torch
import transformers

from glp import flow_matching

# =========================
#   Diffusion Functions
# =========================
def postprocess_on_manifold_wrapper(model, u=0.5, num_timesteps=20, layer_idx=None):
    scheduler = model.scheduler
    num_train_timesteps = scheduler.config.num_train_timesteps
    scheduler.set_timesteps(num_timesteps)
    def postprocess_on_manifold(acts_edit):
        has_seq_dim = len(acts_edit.shape) == 3
        b = acts_edit.shape[0]
        latents = acts_edit
        if has_seq_dim:
            latents = einops.rearrange(latents, "b s d -> (b s) 1 d")
        else:
            latents = einops.rearrange(latents, "b d -> b 1 d")
        latents = model.normalizer.normalize(latents, layer_idx=layer_idx)
        noise = torch.randn_like(latents)
        noisy_latents, _, timesteps, _ = flow_matching.fm_prepare(
            scheduler, 
            latents,
            noise,
            u=torch.ones(latents.shape[0]) * u,
        )
        latents = flow_matching.sample_on_manifold(
            model,
            noisy_latents,
            start_timestep=timesteps[0].item(),
            num_timesteps=num_timesteps,
            layer_idx=layer_idx
        )
        latents = model.normalizer.denormalize(latents, layer_idx=layer_idx)
        if has_seq_dim:
            latents = einops.rearrange(latents, "(b s) 1 d -> b s d", b=b)
        else:
            latents = einops.rearrange(latents, "b 1 d -> b d")
        latents = latents.to(device=acts_edit.device, dtype=acts_edit.dtype)
        return latents
    return postprocess_on_manifold

# =========================
#    Steering Functions
# =========================
def addition_intervention(w=None, alphas=None, postprocess_fn=None):
    if postprocess_fn is None:
        postprocess_fn = lambda x: x
    def rep_act(output, layer_name, inputs):
        nonlocal w, alphas
        use_tuple = isinstance(output, tuple)
        act = output[0] if use_tuple else output
        if w is not None:
            # move to device
            w = w.to(device=act.device, dtype=act.dtype)
            alphas = alphas.to(device=act.device, dtype=act.dtype)
            # reshape based on if batched / unbatched
            if w.ndim == 1:
                w = w[None, None, :]
            elif w.ndim == 2:
                w = w[:, None, :]
            if alphas.ndim == 1:
                alphas = alphas[:, None, None]
            # only apply to every new generated token
            act[:, [-1], :] = postprocess_fn(act[:, [-1], :] + alphas * w)
        return (act, *output[1:]) if use_tuple else act
    return rep_act

def steer_sweep(hf_model, hf_tokenizer, prefixes, alphas, w, layer, act_norm, postprocess_fn=None, batch_size=100, max_new_tokens=20, temperature=0, seed=42):
    hf_tokenizer.padding_side = "left"
    rows = []
    for i in range(0, len(prefixes), batch_size):
        batch = prefixes[i:i + batch_size]
        batch_text = sum([[text] * len(alphas) for text in batch], [])
        batch_alphas = alphas.repeat(len(batch))
        generate_with_intervention = generate_with_intervention_wrapper(seed=seed)
        gen_text = generate_with_intervention(
            batch_text,
            hf_model,
            hf_tokenizer,
            layers=[f"model.layers.{layer}"],
            intervention_wrapper=addition_intervention,
            # save interpretable alpha but apply act_norm
            intervention_kwargs={"w": w, "alphas": batch_alphas * act_norm, "postprocess_fn": postprocess_fn},
            generate_kwargs={"max_new_tokens": max_new_tokens, "do_sample": temperature > 0, "temperature": temperature},
        )
        rows += [{"alpha": a.item(), "prefix": p, "text": t} for a, p, t in zip(batch_alphas, batch_text, gen_text)]
    return pd.DataFrame(rows)

def generate(model, processor, inputs, remove_input=True, **generate_kwargs):
    with torch.no_grad():
        output = model.generate(**inputs, **generate_kwargs)
        if remove_input:
            input_len = inputs["input_ids"].shape[1]
            output = output[:, input_len:]
        output = processor.batch_decode(output, skip_special_tokens=True)
    return output
    
def generate_with_intervention_wrapper(seed=42):
    def generate_with_intervention(text, hf_model, hf_processor, generate_kwargs={"max_new_tokens": 10}, layers=[], intervention_wrapper=None, intervention_kwargs={}, forward_only=False):
        if seed is not None:
            transformers.set_seed(seed)
        inputs = hf_processor(text, return_tensors="pt", padding=True).to(hf_model.device)
        if intervention_wrapper is not None:
            intervention_fn = intervention_wrapper(**intervention_kwargs)
        else:
            intervention_fn = None
        with TraceDict(hf_model, layers=layers, edit_output=intervention_fn) as ret:
            if forward_only:
                outputs = hf_model(**inputs)
                output_text = hf_processor.batch_decode(outputs.logits.argmax(dim=-1), skip_special_tokens=True)
            else:
                output_text = generate(hf_model, hf_processor, inputs, **generate_kwargs)
        return output_text
    return generate_with_intervention
from dataclasses import dataclass, field
import glob
import os

from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
from safetensors import safe_open
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from glp.denoiser import load_glp
from glp.utils_judge import judge_parquets
from glp.script_steer import postprocess_on_manifold_wrapper, steer_sweep

@dataclass
class RunConfig:
    model_name: str = "meta-llama/Llama-3.1-8B"
    weights_folder: str | None = "generative-latent-prior/glp-llama8b-d6"
    ckpt_name: str = "final"
    # SAE decoder directions: an ~8MB slice of the 500 features, row-aligned with feature_file
    dirs_repo: str | None = "generative-latent-prior/llama8b-layer15-llamascope-500"
    dirs_file: str = "llamascope_500_dirs.safetensors"
    # full LlamaScope SAE (residual stream, layer 15, 32x expansion); used when dirs_repo is null, e.g. for custom feature files
    sae_repo: str = "fnlp/Llama3_1-8B-Base-LXR-32x"
    sae_file: str = "Llama3_1-8B-Base-L15R-32x/checkpoints/final.safetensors"
    # data (see README for provenance)
    feature_file: str | None = None  # null downloads the default 500 features from dirs_repo
    prompt_file: str = "cached_inputs/alpaca_eval.csv"
    save_folder: str = "runs/sae"
    # steering
    num_features: int | None = None  # subsample the feature file for a quick pass
    num_prompts: int = 5
    alphas: list[float] = field(default_factory=lambda: [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    act_norm: float | None = 11.6066  # mean layer-15 activation norm of Llama-3.1-8B on FineWeb; null estimates it from the GLP normalizer stats
    max_new_tokens: int = 128
    temperature: float = 1.0
    batch_size: int = 50
    seed: int = 42
    # GLP post-processing
    u: float = 0.5
    num_timesteps: int = 20
    dtype: str = "bfloat16"
    # judge
    judge_model: str = "gpt-4o-mini"
    max_concurrency: int = 100
    overwrite_judge: bool = False
    # plots
    annotate: bool = False

# =========================
#         Steering
# =========================
def sae_steering(config, device="cuda:0"):
    feature_file = config.feature_file or hf_hub_download(config.dirs_repo, "llamascope_500.parquet", repo_type="dataset")
    features = pd.read_parquet(feature_file)
    if config.num_features:
        features = features.head(config.num_features)
    features = features.reset_index(drop=True)
    methods = ["sae"] + (["glp"] if config.weights_folder else [])
    if all(os.path.exists(f"{config.save_folder}/{i}/{m}.parquet") for i in features["index"] for m in methods):
        print(f"Skipping {config.save_folder}")
        return

    # GLP; its tracedict_config records the layer it was trained on (model.layers.15, output)
    glp = load_glp(config.weights_folder, device=device, checkpoint=config.ckpt_name) if config.weights_folder else None
    tracedict_config = dict(glp.tracedict_config) if glp is not None else {"layer_prefix": "model.layers", "layers": [15], "retain": "output"}
    layer = tracedict_config["layers"][0]

    # SAE decoder directions, one row per feature; the packaged slice is row-aligned with the default feature_file
    if config.dirs_repo:
        dirs_path = hf_hub_download(config.dirs_repo, config.dirs_file, repo_type="dataset")
        with safe_open(dirs_path, framework="pt") as f:
            dirs = f.get_tensor("W_dec").float()
        assert len(dirs) >= len(features), f"{config.dirs_file} has {len(dirs)} rows for {len(features)} features; set dirs_repo=null to slice the full checkpoint for a custom feature_file"
    else:
        sae_path = hf_hub_download(config.sae_repo, config.sae_file)
        with safe_open(sae_path, framework="pt") as f:
            W_dec = f.get_tensor("decoder.weight").float()
        dirs = W_dec.T[features["index"].tolist()]

    # sanity check act_norm against an estimate from the GLP's FineWeb statistics
    est_act_norm = torch.sqrt((glp.normalizer.mean.float() ** 2 + glp.normalizer.var.float()).sum()).item() if glp is not None else None
    act_norm = config.act_norm or est_act_norm
    assert act_norm, "act_norm must be set when weights_folder is null"
    if est_act_norm:
        print(f"act_norm {act_norm:.4f} (GLP normalizer estimate {est_act_norm:.4f})")

    hf_tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    hf_tokenizer.pad_token = hf_tokenizer.eos_token
    hf_model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=getattr(torch, config.dtype)).to(device).eval()

    prompt_pool = pd.read_csv(config.prompt_file)["prompt"]
    methods = {"sae": None}
    if glp is not None:
        methods["glp"] = postprocess_on_manifold_wrapper(glp, u=config.u, num_timesteps=config.num_timesteps)

    alphas = torch.tensor(list(config.alphas))
    steer_kwargs = dict(layer=layer, act_norm=act_norm, batch_size=config.batch_size, max_new_tokens=config.max_new_tokens, temperature=config.temperature, seed=config.seed)
    for row, feature in features.iterrows():
        w = dirs[row]
        w = w / torch.linalg.norm(w)
        # each feature gets its own random prompts, seeded by feature index
        prompts = prompt_pool.sample(config.num_prompts, random_state=feature["index"]).tolist()
        for method, postprocess_fn in methods.items():
            save_file = f"{config.save_folder}/{feature['index']}/{method}.parquet"
            if os.path.exists(save_file):
                print(f"Skipping {save_file}")
                continue
            print(f"Steering feature {feature['index']} ({feature['text'][:60]}) with {method}")
            results = steer_sweep(hf_model, hf_tokenizer, prompts, alphas, w, postprocess_fn=postprocess_fn, **steer_kwargs)
            results["concept"] = feature["text"]
            results["method"] = method
            results["feature_index"] = feature["index"]
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            results.to_parquet(save_file)
    os.makedirs(config.save_folder, exist_ok=True)
    OmegaConf.save(config, f"{config.save_folder}/config.yaml")

# =========================
#          Plots
# =========================

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
    "figure.titlesize": 22,
})

STYLES = {
    "sae": dict(color="#BCBD45", label="SAE"),
    "glp": dict(color="tab:pink", label="+Ours"),
}

def paired_bootstrap_ci(df, n_bootstraps=10000, ci=95, seed=0, chunk_size=500):
    """95% bootstrap CI of the per-alpha mean (fluency, concept), resampling generations paired across alphas (non-standard: one resample of generation indices is reused at every alpha)."""
    alphas = np.sort(df["alpha"].unique())
    groups = [df.loc[df["alpha"] == a, ["fluency_llm_score", "concept_llm_score"]].to_numpy(dtype=float) for a in alphas]
    n_tasks = max(len(g) for g in groups)
    data = np.full((len(alphas), n_tasks, 2), np.nan)
    for i, g in enumerate(groups):
        data[i, : len(g)] = g

    rng = np.random.default_rng(seed)
    bootstrapped_means = []
    for start in range(0, n_bootstraps, chunk_size):
        indices = rng.integers(0, n_tasks, size=(min(chunk_size, n_bootstraps - start), n_tasks))
        bootstrapped_means.append(np.nanmean(data[:, indices], axis=2))
    bootstrapped_means = np.concatenate(bootstrapped_means, axis=1)

    lower_p = (100 - ci) / 2
    ci_lower = np.percentile(bootstrapped_means, lower_p, axis=1)
    ci_upper = np.percentile(bootstrapped_means, 100 - lower_p, axis=1)
    return alphas, ci_lower, ci_upper, np.nanmean(data, axis=1)

def plot_tradeoff(config, ax=None, title="500 LlamaScope SAE Concepts"):
    files = sorted(glob.glob(f"{config.save_folder}/*/*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files])
    summary = df.groupby(["method", "alpha"])[["fluency_llm_score", "concept_llm_score"]].mean().reset_index()
    summary.to_csv(f"{config.save_folder}/summary.csv", index=False)

    save_fig = ax is None
    if save_fig:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    for method in [m for m in STYLES if m in df["method"].unique()]:
        style = STYLES[method]
        alphas, ci_lower, ci_upper, means = paired_bootstrap_ci(df[df["method"] == method], seed=config.seed)
        ax.plot(means[:, 0], means[:, 1], linestyle="-", marker="o", alpha=1.0, **style)
        xs = np.concatenate([ci_lower[:, 0], ci_upper[::-1, 0]])
        ys = np.concatenate([ci_lower[:, 1], ci_upper[::-1, 1]])
        ax.fill(xs, ys, color=style["color"], alpha=0.18, linewidth=1, zorder=2)
        if config.annotate:
            for alpha, (f, c) in zip(alphas, means):
                ax.annotate(f"{alpha:.1f}", (f, c), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=7, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel(r"Fluency Score $\uparrow$")
    ax.set_ylabel(r"Concept Score $\uparrow$")
    ax.legend(loc="lower left")
    ax.grid(True, linestyle="--", alpha=0.6)
    print(summary)
    if save_fig:
        plt.tight_layout()
        plt.savefig(f"{config.save_folder}/tradeoff.png")
        plt.close(fig)
        print(f"Saved {config.save_folder}/tradeoff.png")

# =========================
#           Main
# =========================
def main(device="cuda:0"):
    default_config = OmegaConf.structured(RunConfig)
    OmegaConf.set_struct(default_config, False)
    config = OmegaConf.merge(default_config, OmegaConf.from_cli())
    assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY is not set (needed for the LLM judge)"

    sae_steering(config, device=device)
    judge_parquets(sorted(glob.glob(f"{config.save_folder}/**/*.parquet", recursive=True)), model=config.judge_model, max_concurrency=config.max_concurrency, overwrite=config.overwrite_judge)
    plot_tradeoff(config)

if __name__ == "__main__":
    main()

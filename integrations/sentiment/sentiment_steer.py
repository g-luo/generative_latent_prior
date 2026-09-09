from dataclasses import dataclass, field
import glob
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from glp.denoiser import load_glp
from glp.utils_acts import save_acts
from glp.utils_judge import judge_parquets
from glp.script_steer import postprocess_on_manifold_wrapper, steer_sweep

@dataclass
class RunConfig:
    model_name: str = "meta-llama/Llama-3.2-1B"
    weights_folders: list[str] = field(default_factory=lambda: [f"generative-latent-prior/glp-llama1b-d{d}" for d in [3, 6, 12, 24]])
    ckpt_names: list[str] = field(default_factory=lambda: ["final"])  # e.g. ["epoch_0128", "final"] to also steer intermediate checkpoints
    # data
    sst5_train: str = "cached_inputs/sst5_train.csv"
    sst5_test: str | None = "cached_inputs/sst5_test.csv"  # held-out AUROC check
    prefix_file: str = "cached_inputs/openwebtext_neutral_100_prefixes.csv"
    save_root: str = "runs/sentiment"
    # steering
    tasks: list[str] = field(default_factory=lambda: ["positive_sentiment", "negative_sentiment"])
    methods: list[str] = field(default_factory=lambda: ["diffmean", "glp"])
    alphas: list[float] = field(default_factory=lambda: [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    act_norm: float | None = 5.2087  # mean layer-7 activation norm of Llama-3.2-1B on FineWeb (Llama-3.1-8B layer 15: 11.6066); null estimates it from the GLP normalizer stats
    max_new_tokens: int = 20
    temperature: float = 0
    batch_size: int = 100
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
    plot_tradeoffs: bool = False  # per-size sentiment score vs NLL curves

# =========================
#         Steering
# =========================
def get_diffmean_dir(X_train, y_train, l2_normalize=True, eps=1e-8):
    w = X_train[y_train == 1].mean(dim=0) - X_train[y_train == 0].mean(dim=0)
    if l2_normalize:
        w = w / (torch.linalg.norm(w) + eps)
    return w

def sentiment_steering(config, weights_folder, ckpt_name, save_folder, device="cuda:0"):
    if all(os.path.exists(f"{save_folder}/{t}/{m}.parquet") for t in config.tasks for m in config.methods):
        print(f"Skipping {save_folder}")
        return

    # GLP; its tracedict_config records the layer it was trained on (e.g. model.layers.7, output)
    # intermediate checkpoints live under checkpoints/ in the released repos
    checkpoint = ckpt_name if ckpt_name == "final" else f"checkpoints/{ckpt_name}"
    glp = load_glp(weights_folder, device=device, checkpoint=checkpoint) if weights_folder else None
    tracedict_config = dict(glp.tracedict_config) if glp is not None else {"layer_prefix": "model.layers", "layers": [7], "retain": "output"}
    layer = tracedict_config["layers"][0]

    # sanity check act_norm against an estimate from the GLP's FineWeb statistics
    est_act_norm = torch.sqrt((glp.normalizer.mean.float() ** 2 + glp.normalizer.var.float()).sum()).item() if glp is not None else None
    act_norm = config.act_norm or est_act_norm
    assert act_norm, "act_norm must be set when weights_folder is null"
    if est_act_norm:
        print(f"act_norm {act_norm:.4f} (GLP normalizer estimate {est_act_norm:.4f})")

    hf_tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    hf_tokenizer.pad_token = hf_tokenizer.eos_token
    hf_model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=getattr(torch, config.dtype)).to(device).eval()

    # steering vector: diff-of-means of last-token activations on the SST-5 train split
    def last_token_acts(texts):
        acts = save_acts(hf_model, hf_tokenizer, texts, tracedict_config, padding_side="right", token_idx="last", batch_size=32)
        return acts[:, 0, :].float()

    df_train = pd.read_csv(config.sst5_train)
    X_train, y_train = last_token_acts(df_train["prompt"].tolist()), torch.tensor(df_train["target"].values)
    w = get_diffmean_dir(X_train, y_train)
    print(f"Layer {layer}; mean SST-5 activation norm {X_train.norm(dim=-1).mean():.4f}")
    if config.sst5_test and os.path.exists(config.sst5_test):
        df_test = pd.read_csv(config.sst5_test)
        X_test = last_token_acts(df_test["prompt"].tolist())
        print(f"Held-out AUROC of the diff-of-means direction: {roc_auc_score(df_test['target'].values, (X_test @ w).numpy()):.4f}")
    os.makedirs(save_folder, exist_ok=True)
    torch.save(w, f"{save_folder}/sst5_diffmean_layer{layer}.pt")

    # steer
    prefixes = pd.read_csv(config.prefix_file)["prompt"].tolist()
    methods = {m: None for m in config.methods}
    if "glp" in methods:
        assert glp is not None, "weights_folder is required for the glp method"
        methods["glp"] = postprocess_on_manifold_wrapper(glp, u=config.u, num_timesteps=config.num_timesteps)
    steer_kwargs = dict(layer=layer, act_norm=act_norm, batch_size=config.batch_size, max_new_tokens=config.max_new_tokens, temperature=config.temperature, seed=config.seed)
    for task in config.tasks:
        sign = -1 if "negative" in task else 1
        alphas = torch.tensor(list(config.alphas)) * sign
        for method, postprocess_fn in methods.items():
            save_file = f"{save_folder}/{task}/{method}.parquet"
            if os.path.exists(save_file):
                print(f"Skipping {save_file}")
                continue
            print(f"Steering {task} with {method}")
            results = steer_sweep(hf_model, hf_tokenizer, prefixes, alphas, w, postprocess_fn=postprocess_fn, **steer_kwargs)
            results["concept"] = task.replace("_", " ")
            results["method"] = method
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            results.to_parquet(save_file)
    run_config = OmegaConf.merge(config, {"weights_folder": weights_folder, "ckpt_name": ckpt_name, "num_params": sum(p.numel() for p in glp.parameters()) if glp else None})
    OmegaConf.save(run_config, f"{save_folder}/config.yaml")

# =========================
#         Grading
# =========================
SST5_LABELS = ["very negative", "negative", "neutral", "positive", "very positive"]

def convert_sst5_to_score(pred):
    # expected value of the five-point SST-5 classifier (1 = very negative, 5 = very positive)
    labels = [x["label"] for x in pred]
    assert labels == SST5_LABELS
    score = sum(x["score"] * (i + 1) for i, x in enumerate(pred))
    return labels[int(np.argmax([x["score"] for x in pred]))], score

@torch.no_grad()
def conditional_nll(hf_model, hf_tokenizer, prefixes, completions, batch_size=32, max_length=1024):
    # per-token NLL of the completion given the prefix as context, under the steered LLM
    padding_side = hf_tokenizer.padding_side
    hf_tokenizer.padding_side = "right"
    if hf_tokenizer.pad_token is None:
        hf_tokenizer.pad_token = hf_tokenizer.eos_token
    losses = []
    data = list(zip(prefixes, completions))
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        inputs = hf_tokenizer([f"{p} {c}" for p, c in batch], return_tensors="pt", padding="longest", truncation=True, max_length=max_length)
        inputs_prefix = hf_tokenizer([p for p, c in batch], return_tensors="pt", padding="longest", add_special_tokens=False)
        input_ids = inputs["input_ids"].to(hf_model.device)
        attention_mask = inputs["attention_mask"].to(hf_model.device)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # mask padding
        prefix_lengths = inputs_prefix["attention_mask"].sum(dim=1).to(hf_model.device) + 1  # add 1 for bos
        seq_ids = torch.arange(input_ids.shape[1], device=hf_model.device)
        labels[seq_ids < prefix_lengths.unsqueeze(1)] = -100  # mask prefix
        logits = hf_model(input_ids=input_ids, attention_mask=attention_mask).logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        token_losses = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(shift_labels.shape)
        losses.append(token_losses.sum(dim=-1) / ((shift_labels != -100).sum(dim=-1) + 1e-9))
    hf_tokenizer.padding_side = padding_side
    return torch.cat(losses, dim=0)

def grade_parquets(config, files, device="cuda:0"):
    # SST-5 sentiment classifier score and NLL of the continuation
    classifier, hf_model, hf_tokenizer = None, None, None
    for f in files:
        df = pd.read_parquet(f)
        if {"sentiment_label", "sentiment_score", "nll"} <= set(df.columns):
            print(f"Skipping grading for {f}")
            continue
        if classifier is None:
            classifier = pipeline(model="SetFit/distilbert-base-uncased__sst5__all-train", device=device)
            hf_tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            hf_tokenizer.pad_token = hf_tokenizer.eos_token
            hf_model = AutoModelForCausalLM.from_pretrained(config.model_name).to(device).eval()
        print(f"Grading {f}")
        texts = df["text"].tolist()
        preds = sum([classifier(texts[i:i + 256], return_all_scores=True) for i in range(0, len(texts), 256)], [])
        df["sentiment_label"], df["sentiment_score"] = zip(*[convert_sst5_to_score(p) for p in preds])
        df["nll"] = conditional_nll(hf_model, hf_tokenizer, df["prefix"].tolist(), texts).float().cpu().numpy()
        df.to_parquet(f)

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
    "legend.fontsize": 13,
    "figure.titlesize": 22,
})

STYLES = {
    "diffmean": dict(color="tab:orange", label="DiffMean"),
    "glp": dict(color="tab:blue", label="+Ours"),
}

def load_summary(save_folder):
    files = sorted(glob.glob(f"{save_folder}/*/*.parquet"))
    df = pd.concat([pd.read_parquet(f).assign(task=f.split("/")[-2], method=os.path.basename(f).split(".")[0]) for f in files])
    summary = df.groupby(["task", "method", "alpha"])[["fluency_llm_score", "concept_llm_score"]].mean().reset_index()
    summary.to_csv(f"{save_folder}/summary.csv", index=False)
    return df, summary

def plot_tradeoff_panel(ax, df, title):
    # sentiment-classifier score vs NLL, one point per steering
    # coefficient, with 95% bootstrap CI bands
    polarity = "Negative" if "negative" in title.lower() else "Positive"
    all_means = []
    for method in [m for m in STYLES if m in df["method"].unique()]:
        rows = df[df["method"] == method].copy()
        if polarity == "Negative":  # report sentiment toward the steered concept
            rows["sentiment_score"] = 6 - rows["sentiment_score"]
        groups = sorted(rows.groupby("alpha"), key=lambda group: abs(group[0]))
        means = np.array([group[["nll", "sentiment_score"]].mean().values for _, group in groups])
        lo, hi = (np.array(x) for x in zip(*[bootstrap_ci(group[["nll", "sentiment_score"]].values) for _, group in groups]))
        ax.plot(means[:, 0], means[:, 1], marker="o", linestyle="-", **STYLES[method])
        ax.fill(np.concatenate([lo[:, 0], hi[::-1, 0]]), np.concatenate([lo[:, 1], hi[::-1, 1]]), color=STYLES[method]["color"], alpha=0.18, linewidth=1, zorder=2)
        all_means.append(means)
    # default ylim (2.8, 5), extended if the data falls outside it, with 2.5% padding
    all_means = np.concatenate(all_means)
    y0, y1 = min(2.8, all_means[:, 1].min()), max(5, all_means[:, 1].max())
    ax.set_ylim(y0 - 0.025 * (y1 - y0), y1 + 0.025 * (y1 - y0))
    ax.invert_xaxis()
    ax.set_title(title)
    ax.set_xlabel(r"Negative Log Likelihood $\downarrow$")
    ax.set_ylabel(rf"{polarity} Sentiment Score $\uparrow$")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.6)

def bootstrap_ci(values, n_bootstraps=10000, ci=95, seed=0):
    # per-column 95% bootstrap CI of the mean
    values = np.asarray(values, dtype=float)
    idx = np.random.default_rng(seed).integers(0, values.shape[0], size=(n_bootstraps, values.shape[0]))
    means = values[idx].mean(axis=1)
    return np.percentile(means, (100 - ci) / 2, axis=0), np.percentile(means, 100 - (100 - ci) / 2, axis=0)

def plot_tradeoff(config, save_folder):
    # one panel per steered concept
    df, _ = load_summary(save_folder)
    tasks = sorted(df["task"].unique(), reverse=True)  # positive sentiment first
    fig, axs = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 6), dpi=300)
    for ax, task in zip(np.atleast_1d(axs), tasks):
        plot_tradeoff_panel(ax, df[df["task"] == task], f"Concept: {task.replace('_', ' ').title()}")
    plt.tight_layout()
    plt.savefig(f"{save_folder}/tradeoff.png")
    plt.savefig(f"{save_folder}/tradeoff.pdf")
    plt.close(fig)
    print(f"Saved {save_folder}/tradeoff.png")

def plot_scaling(config):
    # mean of concept and fluency over |alpha| > 1 vs training FLOPs (6ND)
    rows = []
    for folder in sorted(glob.glob(f"{config.save_root}/*/")):
        run_config = OmegaConf.load(f"{folder}/config.yaml")
        if run_config.num_params is None:
            continue
        _, summary = load_summary(folder)
        # positive sentiment only, mean of the two scores over |alpha| > 1
        stats = summary[(summary["task"] == "positive_sentiment") & (summary["method"] == "glp") & (summary["alpha"].abs() > 1)]
        score = stats[["fluency_llm_score", "concept_llm_score"]].mean(axis=1).mean()
        # acts seen by the checkpoint, from its name ("final" = epoch_1024, 1M acts per epoch)
        ckpt = str(run_config.ckpt_name)
        train_acts = 1.024e9 if ckpt == "final" else int(ckpt.split("_")[-1]) * 1e6
        rows.append({"folder": folder, "num_params": run_config.num_params, "flops": 6 * run_config.num_params * train_acts, "score": score})
    scaling = pd.DataFrame(rows).sort_values(["num_params", "flops"])
    scaling.to_csv(f"{config.save_root}/scaling.csv", index=False)
    # one Blues shade per GLP size; a size becomes a curve once it
    # has multiple checkpoints, and a lone final checkpoint stays a disconnected point
    sizes = sorted(scaling["num_params"].unique())
    colors = plt.cm.Blues(np.linspace(0.4, 1, len(sizes)))
    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
    for color, size in zip(colors, sizes):
        group = scaling[scaling["num_params"] == size].sort_values("flops")
        if len(group) > 1:
            ax.plot(group["flops"], group["score"], marker=".", markersize=5, linewidth=1.5, linestyle="-", color=color, label=f"{size / 1e9:.1f}B")
        else:
            ax.plot(group["flops"], group["score"], marker="o", markersize=5, linestyle="None", color=color, label=f"{size / 1e9:.1f}B")
    ax.set_xscale("log")
    ax.set_xlim(10 ** np.floor(np.log10(scaling["flops"].min())), 10 ** np.ceil(np.log10(scaling["flops"].max())))
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_ylim(0.1, 0.7)
    ax.set_xlabel("FLOPs")
    ax.set_ylabel(r"Concept \& Fluency Mean")
    ax.set_title("On-Manifold Sentiment Steering")
    ax.legend(title="GLP Size", loc="lower right", title_fontsize=plt.rcParams["legend.fontsize"])
    ax.grid(True, which="major", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{config.save_root}/scaling.png")
    plt.savefig(f"{config.save_root}/scaling.pdf")
    plt.close(fig)
    print(scaling)
    print(f"Saved {config.save_root}/scaling.png")

# =========================
#           Main
# =========================
def main(device="cuda:0"):
    default_config = OmegaConf.structured(RunConfig)
    OmegaConf.set_struct(default_config, False)
    config = OmegaConf.merge(default_config, OmegaConf.from_cli())
    assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY is not set (needed for the LLM judge)"

    # the "final" checkpoint keeps the original per-size folder; other checkpoints get a suffix
    save_folders = []
    for weights_folder in config.weights_folders:
        for ckpt_name in config.ckpt_names:
            suffix = "" if ckpt_name == "final" else f"-{ckpt_name}"
            save_folder = f"{config.save_root}/{weights_folder.split('/')[-1]}{suffix}"
            save_folders.append((weights_folder, ckpt_name, save_folder))
    for weights_folder, ckpt_name, save_folder in save_folders:
        sentiment_steering(config, weights_folder, ckpt_name, save_folder, device=device)
    judge_parquets(sorted(glob.glob(f"{config.save_root}/**/*.parquet", recursive=True)), model=config.judge_model, max_concurrency=config.max_concurrency, overwrite=config.overwrite_judge)
    grade_parquets(config, sorted(glob.glob(f"{config.save_root}/**/*.parquet", recursive=True)), device=device)
    if config.plot_tradeoffs:
        for _, _, save_folder in save_folders:
            plot_tradeoff(config, save_folder)
    if len(save_folders) > 1:
        plot_scaling(config)

if __name__ == "__main__":
    main()

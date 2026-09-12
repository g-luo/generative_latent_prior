from baukit import TraceDict
from collections import defaultdict
from dataclasses import dataclass
import einops
import glob
import json
import numpy as np
from omegaconf import OmegaConf
import os
import pandas as pd
from scipy.stats import bootstrap
from sklearn.preprocessing import LabelEncoder
import subprocess
import torch
from tqdm import tqdm

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

from glp.denoiser import load_glp

# ====================================
#    Logistic Regression Functions
# ====================================
# adapted from SAE-Probes (Kantamneni et al., 2025) `find_best_reg`, which cross-validates an L2 logistic regression over C
# and reports val / test AUC; we use sklearn's LogisticRegressionCV with standardized inputs and a fixed grid of Cs:
# https://github.com/JoshEngels/SAE-Probes/blob/9917d94e8ab568e3686b6d3801fab52cda54b81d/utils_training.py#L59-L135
def run_sklearn_logreg(X_train, y_train, X_test, y_test, parallel=True, n_jobs=-1, Cs=None, seed=1, max_iter=1000):
    if Cs is None:
        Cs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    # standardize to use fixed Cs
    # use k-fold cross val since train is small
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(
            Cs=Cs, 
            cv=5,
            scoring='roc_auc',
            max_iter=max_iter,
            random_state=seed,
            n_jobs=n_jobs,
            penalty='l2'
        )
    )
    pipe.fit(X_train, y_train)
    metrics = {}
    # final test auc
    y_test_pred_proba = pipe.predict_proba(X_test)[:, 1]
    metrics['test_auc'] = roc_auc_score(y_test, y_test_pred_proba)
    # avg val auc over all folds
    best_c_idx = np.where(np.array(pipe['logisticregressioncv'].Cs_) == pipe['logisticregressioncv'].C_[0])[0][0]
    metrics['val_auc'] = pipe['logisticregressioncv'].scores_[1].mean(axis=0)[best_c_idx]
    return metrics

def run_sklearn_logreg_batched(X_train, y_train, X_test, y_test, device=None, **kwargs):
    assert len(X_train.shape) == len(X_test.shape) == 3
    assert len(y_train.shape) == len(y_test.shape) == 1
    pt_to_np = lambda x: x.detach().cpu().numpy() if torch.is_tensor(x) else x
    X_train, y_train, X_test, y_test = pt_to_np(X_train), pt_to_np(y_train), pt_to_np(X_test), pt_to_np(y_test)
    metrics_batch = Parallel(n_jobs=-1)(delayed(run_sklearn_logreg)(X_train[i], y_train, X_test[i], y_test, **kwargs) for i in range(X_train.shape[0]))
    # metrics_batch = [run_sklearn_logreg(X_train[i], y_train, X_test[i], y_test, **kwargs) for i in range(X_train.shape[0])]
    metrics_batch = {k: [m[k] for m in metrics_batch] for k in metrics_batch[0].keys()}
    return np.array(metrics_batch["val_auc"]), np.array(metrics_batch["test_auc"])

def prefilter_and_reshape_to_oned(X_train_batched, X_test_batched, y_train, device, topk=512):
    if topk == -1:
        topk = X_train_batched.shape[-1]
        print(f"Using all dimensions: {topk}")
    b, _, d = X_train_batched.shape
    X_train = einops.rearrange(X_train_batched, "b n d -> n (b d)")
    X_test = einops.rearrange(X_test_batched, "b n d -> n (b d)")
    # copied from SAE-Probes `get_sorted_indices` for ranking by class mean difference:
    # https://github.com/JoshEngels/SAE-Probes/blob/9917d94e8ab568e3686b6d3801fab52cda54b81d/train_sae_probes.py#L79-L82
    X_train_diff = X_train[y_train == 1].mean(dim=0) - X_train[y_train == 0].mean(dim=0)
    sorted_indices = torch.argsort(torch.abs(X_train_diff), descending=True)
    sorted_indices = sorted_indices[:topk]
    X_train = X_train[:, sorted_indices][None, ...]
    X_test = X_test[:, sorted_indices][None, ...]
    X_train = einops.rearrange(X_train, "1 n d -> d n 1")
    X_test = einops.rearrange(X_test, "1 n d -> d n 1")
    top_batch_idxs = sorted_indices
    return X_train, X_test, top_batch_idxs.tolist()

# =========================
#   Diffusion Functions
# =========================
def get_meta_neurons_wrapper(seed=42):
    generator = torch.Generator().manual_seed(seed)
    @torch.no_grad()
    def get_meta_neurons(model, X, device, u, layers, layer_idx=None, batch_size=None):
        batch_size = batch_size or X.shape[0]
        # normalize and reshape X
        latents = X.to(device)
        latents = latents[:, None, :]
        latents = model.normalizer.normalize(latents, layer_idx=layer_idx)
        all_ret = []
        for i in range(0, latents.shape[0], batch_size):
            # collect diffusion model internals
            with TraceDict(model, layers=[x[0] for x in layers], retain_input=True, retain_output=True) as ret:
                noise_pred = model(
                    latents=latents[i:i+batch_size],
                    u=u[i:i+batch_size],
                    layer_idx=layer_idx,
                    generator=generator
                )
            ret = [getattr(ret[layer], loc).detach().cpu() for layer, loc in layers]
            # combine all outputs
            ret = torch.stack(ret)
            all_ret.append(ret)
        # ret is shape (num_layers, batch_size, seq_len, dim)
        all_ret = torch.cat(all_ret, dim=1)
        return all_ret
    return get_meta_neurons

def get_meta_neurons_layer_time(model, device, X, u, layers, seed, batch_size=None, diffusion_kwargs={}):
    # add time dim and batchify to "(b u) d"
    u_size = u.shape[0]
    num_samples = X.shape[0]
    X_exp = X[:, None, :].repeat(1, u_size, 1)
    u_exp = u[None, :, :].repeat(num_samples, 1, 1)
    X_exp = einops.rearrange(X_exp, "b u d -> (b u) d")
    u_exp = einops.rearrange(u_exp, "b u 1 -> (b u) 1")
    # add layer dim and batchify to "(u l) b d"
    get_meta_neurons = get_meta_neurons_wrapper(seed=seed)
    X_diffusion = get_meta_neurons(model, X_exp, device, u_exp, layers, batch_size=batch_size, **diffusion_kwargs)
    layer_size, _, dim_size = X_diffusion.shape
    X_diffusion = einops.rearrange(X_diffusion, "l (b u) d -> (l u) b d", b=num_samples, u=u_size)
    return X_diffusion, (layer_size, u_size)

def get_meta_neurons_locations(model, layer_prefix="denoiser.model.layers.{i}.down_proj", location="input"):
    layers = []
    num_layers = len(model.denoiser.model.layers)
    for i in range(num_layers):
        layers.append((layer_prefix.format(i=i), location))
    return layers

# ===================
#   Main Functions     
# ===================
def compile_probe_results(save_folder):
    all_test_aucs = defaultdict(list)
    for file in glob.glob(f"{save_folder}/**/*.json", recursive=True):
        method = "_".join(file.split("/")[-2:]).split(".")[0]
        result = json.load(open(file))
        val_aucs = list(result["val_aucs"].values())
        test_aucs = list(result["test_aucs"].values())
        best_location = np.argmax(val_aucs)
        test_auc = test_aucs[best_location]
        all_test_aucs[method].append(test_auc)

    results = []
    for method, aucs in sorted(all_test_aucs.items()):
        aucs_array = np.array(aucs)
        mean_auc = np.mean(aucs_array)
        res = bootstrap(
            (aucs_array,), 
            np.mean, 
            confidence_level=0.95, 
            n_resamples=10000, 
            method='percentile',
        )
        low_ci = res.confidence_interval.low
        high_ci = res.confidence_interval.high
        results.append({
            "method": method,
            "mean_auc": mean_auc,
            "ci_low": low_ci,
            "ci_high": high_ci,
        })

    results_df = pd.DataFrame(results)
    return results_df

def download_cached_acts(cached_acts_folder, df_folder):
    # the 113 binary classification datasets are from SAE-Probes (Kantamneni et al., 2025): https://github.com/JoshEngels/SAE-Probes
    script = f"""
    if [ ! -d "{df_folder}" ]; then
        mkdir -p {df_folder}
        wget -O {df_folder}.zip "https://www.dropbox.com/scl/fo/lvajx9100jsy3h9cvis7q/ACU8osTw0FCM_X-d8Wn-3ao/cleaned_data?rlkey=tq7td61h1fufm01cbdu2oqsb5&dl=1"
        unzip -o {df_folder}.zip -d {df_folder}
        rm {df_folder}.zip
    fi
    huggingface-cli download generative-latent-prior/{os.path.basename(cached_acts_folder)} \
        --repo-type dataset \
        --local-dir {cached_acts_folder} \
        --local-dir-use-symlinks False
    """
    subprocess.run(script, shell=True, check=True, executable='/bin/bash')

def load_cached_acts(dataset_folder, df_path):
    # the cached train / test indices were generated with SAE-Probes `get_train_test_indices(y, num_train, num_test, pos_ratio=0.5, seed=42)`,
    # where num_train = min(1024, N - 100) and num_test = N - num_train - 1, with N the dataset size after class balancing:
    # https://github.com/JoshEngels/SAE-Probes/blob/9917d94e8ab568e3686b6d3801fab52cda54b81d/utils_data.py#L73-L105
    df_all = pd.read_csv(df_path)
    def load(split):
        indices = json.load(open(f"{dataset_folder}/indices_{split}.json"))
        X = torch.load(f"{dataset_folder}/X_{split}.pt")
        X = X[:, 0, :]
        df = df_all.iloc[indices]
        le = LabelEncoder()
        y = le.fit_transform(df['target'].values)
        y = torch.tensor(y)
        return X, y, df
    X_train, y_train, df_train = load("train")
    X_test, y_test, df_test = load("test")
    return X_train.float(), y_train, X_test.float(), y_test

@dataclass
class ScalarProbingConfig:
    save_folder: str = "runs/scalar_probing"
    cached_acts_folder: str = "data/llama8b-layer15-sae-probes"
    df_folder: str = "data/sae-probes"
    weights_folder: str | None = "generative-latent-prior/glp-llama8b-d6"
    ckpt_name: str | None = "final"
    u: float | None = 0.9
    topk: int | None = 512
    seed: int = 42
    batch_size: int | None = None # set this to a small number if you're getting OOM

def scalar_probing(device="cuda:0"):
    default_config = OmegaConf.structured(ScalarProbingConfig)
    OmegaConf.set_struct(default_config, False)
    config = OmegaConf.merge(default_config, OmegaConf.from_cli())

    if not os.path.exists(config.cached_acts_folder) or not os.path.exists(config.df_folder):
        download_cached_acts(config.cached_acts_folder, config.df_folder)

    u = torch.tensor([config.u])[:, None]
    model = load_glp(config.weights_folder, device=device, checkpoint=config.ckpt_name)
    weights_name = os.path.basename(config.weights_folder)
        
    for dataset_folder in tqdm(sorted(glob.glob(f"{config.cached_acts_folder}/*"))):
        dataset_name = os.path.basename(dataset_folder)
        df_path = f"{config.df_folder}/{dataset_name}.csv"
        X_train, y_train, X_test, y_test = load_cached_acts(dataset_folder, df_path=df_path)
        save_file = f"{config.save_folder}/{dataset_name}/{weights_name}/{config.ckpt_name}.json"

        if os.path.exists(save_file):
            continue

        if config.weights_folder:
            results = {}
            # get diffusion meta-neurons
            layers = get_meta_neurons_locations(model)
            X_train_diffusion, (layer_size, u_size) = get_meta_neurons_layer_time(model, device, X_train, u, layers, config.seed, config.batch_size)
            X_test_diffusion, _ = get_meta_neurons_layer_time(model, device, X_test, u, layers, config.seed, config.batch_size)
            # run prefiltering and reshape for 1-D probing
            X_train_diffusion, X_test_diffusion, top_batch_idxs = prefilter_and_reshape_to_oned(X_train_diffusion, X_test_diffusion, y_train, device, topk=config.topk)
            # run logistic regression
            val_aucs, test_aucs = run_sklearn_logreg_batched(X_train_diffusion, y_train, X_test_diffusion, y_test, device=device)
            # save results
            format_aucs = lambda aucs: {idx: auc.item() for idx, auc in zip(top_batch_idxs, aucs)}
            results["val_aucs"] = format_aucs(val_aucs)
            results["test_aucs"] = format_aucs(test_aucs)
            results["layers"] = layers
            results["u"] = u.flatten().tolist()
        
        results["config"] = config
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        json.dump({k: v for k, v in results.items() if k != "config"} | {"config": OmegaConf.to_container(results["config"], resolve=True)}, open(save_file, "w"))

    results_df = compile_probe_results(config.save_folder)
    results_df.to_csv(f"{config.save_folder}/results.csv", index=False)
    print(results_df)

if __name__ == "__main__":
    scalar_probing()
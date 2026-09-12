import asyncio
from dataclasses import dataclass
import einops
import glob
import json
from omegaconf import OmegaConf
import numpy as np
import os
import pandas as pd
from scipy.stats import bootstrap
import torch

from baukit import TraceDict
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from glp.denoiser import load_glp
from glp.utils_judge import gpt_batch_call

# =======================
#     TopK Tracking
# =======================
class DocsTopKTracker:
    """
    Track the top-k documents per meta-neuron by max token activation.
    """
    def __init__(self, num_neurons, k, device):
        self.k = k
        self.top_values = torch.full((num_neurons, k), -torch.inf, device=device)
        self.top_doc_idxs = torch.full((num_neurons, k), -1, dtype=torch.long, device=device)

    def update(self, doc_idxs, doc_max_acts):
        # doc_max_acts is (docs, neurons); merge with the current top-k and keep the best k
        values = torch.cat([self.top_values, doc_max_acts.T], dim=1)
        idxs = torch.cat([self.top_doc_idxs, doc_idxs.expand(self.top_values.shape[0], -1)], dim=1)
        top = torch.topk(values, self.k, dim=1)
        self.top_values = top.values
        self.top_doc_idxs = torch.gather(idxs, 1, top.indices)

# =======================
#   Meta-Neuron Functions
# =======================
def load_fineweb_docs(config):
    dataset = load_dataset(config.dataset_path, config.dataset_name, split="train")
    np.random.seed(config.docs_seed)
    random_idxs = np.random.permutation(range(len(dataset)))
    train_docs = random_idxs[:-config.val_size]
    return dataset.select(train_docs[:config.num_docs])

def load_models(config, device):
    model = load_glp(config.weights_folder, device=device, checkpoint=config.ckpt_name)
    # meta-neurons are the post-SwiGLU activations of every GLP MLP block (see get_meta_neurons_locations in script_probe.py)
    glp_layers = [f"denoiser.model.layers.{i}.down_proj" for i in range(len(model.denoiser.model.layers))]
    # intervene on the layer the glp was trained on
    llm_layer = f"model.layers.{dict(model.tracedict_config)['layers'][0]}"
    hf_tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    hf_tokenizer.pad_token = hf_tokenizer.eos_token
    hf_tokenizer.padding_side = "right"
    hf_model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=getattr(torch, config.dtype)).to(device).eval()
    return model, glp_layers, llm_layer, hf_model, hf_tokenizer

@torch.no_grad()
def iterate_meta_neurons(config, docs, model, glp_layers, llm_layer, hf_model, hf_tokenizer, device):
    """
    Yield the meta-neuron activations of each batch of documents.
    Noise is seeded per batch so that a second pass sees identical activations.
    """
    for i in range(0, len(docs), config.batch_size):
        batch = docs[i:i + config.batch_size]["text"]
        inputs = hf_tokenizer(batch, return_tensors="pt", padding="max_length", truncation=True, max_length=config.max_length).to(device)
        with TraceDict(hf_model, layers=[llm_layer], retain_output=True) as ret:
            hf_model(**inputs)
        acts = ret[llm_layer].output
        acts = acts[0] if isinstance(acts, tuple) else acts
        # exclude bos and padding
        mask = inputs["attention_mask"].bool()
        mask[:, 0] = False
        # one glp forward at noise level u, same as script_probe.py
        latents = einops.rearrange(acts, "b s d -> (b s) 1 d").float()
        latents = model.normalizer.normalize(latents)
        generator = torch.Generator().manual_seed(config.seed + i)
        u = torch.full((latents.shape[0],), config.u)
        with TraceDict(model, layers=glp_layers, retain_input=True) as ret:
            model(latents=latents, u=u, generator=generator)
        neuron_acts = torch.stack([ret[layer].input for layer in glp_layers])  # (layers, (b s), d_mlp)
        neuron_acts = einops.rearrange(neuron_acts, "l (b s) d -> b s (l d)", b=acts.shape[0]).float()
        neuron_acts[~mask] = -torch.inf
        yield i, batch, inputs, mask, neuron_acts

def track_top_docs(config, device):
    # first pass: top-k documents per meta-neuron by max activation
    model, glp_layers, llm_layer, hf_model, hf_tokenizer = load_models(config, device)
    docs = load_fineweb_docs(config)
    tracker = None
    for i, batch, inputs, mask, neuron_acts in iterate_meta_neurons(config, docs, model, glp_layers, llm_layer, hf_model, hf_tokenizer, device):
        if tracker is None:
            tracker = DocsTopKTracker(neuron_acts.shape[-1], config.top_docs, device)
        doc_idxs = torch.arange(i, i + neuron_acts.shape[0], device=device)
        tracker.update(doc_idxs, neuron_acts.max(dim=1).values)
        print(f"{i + neuron_acts.shape[0]}/{len(docs)} docs", flush=True)
    torch.save({
        "top_values": tracker.top_values.cpu(),
        "top_doc_idxs": tracker.top_doc_idxs.cpu(),
        "num_layers": len(glp_layers),
    }, f"{config.save_folder}/top_doc_idxs_per_neuron.pt")

def gather_top_tokens(config, device):
    # second pass: per-token activations of each (meta-neuron, top document) pair
    topk = torch.load(f"{config.save_folder}/top_doc_idxs_per_neuron.pt")
    top_doc_idxs = topk["top_doc_idxs"].to(device)
    doc_to_neurons = {}
    for neuron, rank in zip(*torch.where(top_doc_idxs != -1)):
        doc_to_neurons.setdefault(top_doc_idxs[neuron, rank].item(), []).append(neuron.item())
    model, glp_layers, llm_layer, hf_model, hf_tokenizer = load_models(config, device)
    docs = load_fineweb_docs(config)
    store = {}
    for i, batch, inputs, mask, neuron_acts in iterate_meta_neurons(config, docs, model, glp_layers, llm_layer, hf_model, hf_tokenizer, device):
        for j in range(neuron_acts.shape[0]):
            doc_idx = i + j
            if doc_idx not in doc_to_neurons:
                continue
            neurons = torch.tensor(doc_to_neurons[doc_idx], device=device)
            store[doc_idx] = {
                "tokens": hf_tokenizer.convert_ids_to_tokens(inputs["input_ids"][j][mask[j]]),
                "neurons": neurons.tolist(),
                "activations": neuron_acts[j][mask[j]][:, neurons].T.cpu().numpy().astype(np.float16),
            }
        print(f"{i + neuron_acts.shape[0]}/{len(docs)} docs", flush=True)
    torch.save(store, f"{config.save_folder}/top_tokens_per_doc.pt")

def build_store(config):
    # join the two passes: per meta-neuron, its top documents with tokens and per-token activations
    topk = torch.load(f"{config.save_folder}/top_doc_idxs_per_neuron.pt")
    store = torch.load(f"{config.save_folder}/top_tokens_per_doc.pt", weights_only=False)
    d_mlp = topk["top_values"].shape[0] // topk["num_layers"]
    rows = []
    for neuron in range(topk["top_values"].shape[0]):
        top_documents = []
        for value, doc_idx in zip(topk["top_values"][neuron].tolist(), topk["top_doc_idxs"][neuron].tolist()):
            if doc_idx == -1:
                continue
            doc = store[doc_idx]
            top_documents.append({
                "doc_idx": doc_idx,
                "max_activation": value,
                "tokens": doc["tokens"],
                "activations": doc["activations"][doc["neurons"].index(neuron)].tolist(),
            })
        rows.append({"layer_id": neuron // d_mlp, "neuron_id": neuron % d_mlp, "top_documents": top_documents})
    pd.DataFrame(rows).to_parquet(f"{config.save_folder}/top_docs_per_neuron.parquet")

def join_probes(config, top_docs=3, bold_frac=0.4):
    # per probing task, the best meta-neuron (by val auc) and its top documents,
    # with the top tokens (>= bold_frac of the doc max) marked in **bold**
    store = pd.read_parquet(f"{config.save_folder}/top_docs_per_neuron.parquet")
    d_mlp = int(store["neuron_id"].max()) + 1
    hf_tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    def render(doc):
        acts = np.array(doc["activations"])
        thr = bold_frac * acts.max()
        out, bold = "", False
        for token, act in zip(doc["tokens"], acts):
            s = hf_tokenizer.convert_tokens_to_string([token]).replace("\n", " ")
            lead, s = s[:len(s) - len(s.lstrip())], s.lstrip()
            if (act >= thr) != bold:
                # open / close a bold span, keeping whitespace outside it
                out += (lead + "**") if not bold else ("**" + lead)
                lead, bold = "", not bold
            out += lead + s
        return out + ("**" if bold else "")
    rows = []
    for file in sorted(glob.glob(f"{config.probe_folder}/**/*.json", recursive=True)):
        result = json.load(open(file))
        val_aucs, test_aucs = result["val_aucs"], result["test_aucs"]
        best = max(val_aucs, key=val_aucs.get)
        neuron = int(best)
        row = {
            "task": file.split("/")[-3],
            "test_auc": test_aucs[best],
            "val_auc": val_aucs[best],
            "layer_id": neuron // d_mlp,
            "neuron_id": neuron % d_mlp,
        }
        for i, doc in enumerate(store.iloc[neuron]["top_documents"][:top_docs]):
            row[f"top_doc_{i + 1}"] = render(doc)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("test_auc", ascending=False)
    df.to_csv(f"{config.save_folder}/probe_to_meta_neurons.csv", index=False, float_format="%.4f")
    print(df[["task", "test_auc", "layer_id", "neuron_id"]].head(10).to_string())

# =======================
#       Autointerp
# =======================

# this system prompt is copied from Llama Scope (He et al., 2024):
# https://github.com/OpenMOSS/Language-Model-SAEs/blob/4dbbc45056e56995099e2184f1e8fc5bdc9378db/src/lm_saes/analysis/autointerp/explanation_prompts.py#L159-L239
AUTOINTERP_SYSTEM_PROMPT = """We're studying features in a neural network. Each feature activates on some particular word/words/substring/concept in a short document. The activating words in each document are indicated with << ... >>. We will give you a list of documents on which the feature activates, in order from most strongly activating to least strongly activating.

Your task is to:

First, Summarize the Activation: Look at the parts of the document the feature activates for and summarize in a single sentence what the feature is activating on. Try not to be overly specific in your explanation. Note that some features will activate only on specific words or substrings, but others will activate on most/all words in a sentence provided that sentence contains some particular concept. Your explanation should cover most or all activating words (for example, don't give an explanation which is specific to a single word if all words in a sentence cause the feature to activate). Pay attention to things like the capitalization and punctuation of the activating words or concepts, if that seems relevant. Keep the explanation as short and simple as possible, limited to 20 words or less. Omit punctuation and formatting. You should avoid giving long lists of words.

Second, Assess Activation Consistency: Based on your summary and the provided examples, evaluate the consistency of the feature's activation. Return your assessment as a single integer from the following scale:

5: Clear pattern with no deviating examples
4: Clear pattern with one or two deviating examples
3: Clear overall pattern but quite a few examples not fitting that pattern
2: Broad consistent theme but lacking structure
1: No discernible pattern

Third, Assess Feature Complexity: Based on your summary and the nature of the activation, evaluate the complexity of the feature. Return your assessment as a single integer from the following scale:

5: Rich feature firing on diverse contexts with an interesting unifying theme, e.g., "feelings of togetherness"
4: Feature relating to high-level semantic structure, e.g., "return statements in code"
3: Moderate complexity, such as a phrase, category, or tracking sentence structure, e.g., "website URLs"
2: Single word or token feature but including multiple languages or spelling, e.g., "mentions of dog"
1: Single token feature, e.g., "the token '('"

Your output should be a JSON object that has the following fields: `steps`, `final_explanation`, `activation_consistency`, `complexity`. `steps` should be an array of strings with a length not exceeding 3, each representing a step in the chain-of-thought process. `final_explanation` should be a string in the form of 'This feature activates on... '. `activation_consistency` should be an integer between 1 and 5, representing the consistency of the feature. `complexity` should be an integer between 1 and 5, representing the complexity of the feature.

Some examples:

{
    "steps": ["Activating token: <<knows>>. Contextual tokens: Who, ?. Pattern: <<knows>> is consistently activated, often found in sentences starting with interrogative words like 'Who' and ending with a question mark.", "Shared features include consistent activation on the word 'knows'. The surrounding text always forms a question. The questions do not seem to expect a literal answer, suggesting they are rhetorical.", "This feature activates on the word knows in rhetorical questions"],
    "final_explanation": "The feature activates on the word 'knows' in rhetorical questions.",
    "activation_consistency": 5,
    "complexity": 4
}

{
    "steps": ["Activating tokens: <<Entwickler>>, <<Enterprise>>, <<Entertainment>>, <<Entity>>, <<Entrance>>. Pattern: All activating instances are on words that begin with the specific substring 'Ent'. The activation is on the 'Ent' portion itself.", "The shared feature across all examples is the presence of words starting with the capitalized substring 'Ent'. The feature appears to be case-sensitive and position-specific (start of the word). No other contextual or semantic patterns are observed."],
    "final_explanation": "The feature activates on the substring 'Ent' at the start of words",
    "activation_consistency": 5,
    "complexity": 1
}

{
    "steps": ["Activating tokens: <<budget deficit>>, <<interest rates>>, <<fiscal stimulus>>, <<trade policy>>, <<unemployment benefits>>. Pattern: Activations highlight phrases and concepts central to economic discussions and government actions.","The examples consistently involve discussions of economic indicators, government spending, financial regulation, or international trade agreements. While most activations clearly relate to economic policies enacted or debated by governmental bodies, some activations might be on broader economic news or expert commentary where the direct link to a specific government policy is less explicit, or on related but not identical topics like corporate financial health in response to policy."],
    "final_explanation": "The feature activates on text about government economic policy",
    "activation_consistency": 3,
    "complexity": 5
}


"""

# this autointerp prompt is adapted from Llama Scope (He et al., 2024):
# https://github.com/OpenMOSS/Language-Model-SAEs/blob/4dbbc45056e56995099e2184f1e8fc5bdc9378db/src/lm_saes/analysis/samples.py#L95-L105
def autointerp_prompt(top_documents, hf_tokenizer, activation_threshold=0.7):
    # documents in order of max activation, with tokens above activation_threshold (min-max normalized) marked <<like this>>
    user_prompt = "The activating documents are given below:\n\n"
    for i, doc in enumerate(top_documents, 1):
        acts = np.array(doc["activations"])
        norm_acts = (acts - acts.min()) / (acts.max() - acts.min() if acts.max() > acts.min() else 1.0)
        text = "".join(f"<<{hf_tokenizer.convert_tokens_to_string([t])}>>" if a >= activation_threshold else hf_tokenizer.convert_tokens_to_string([t]) for t, a in zip(doc["tokens"], norm_acts))
        user_prompt += f"Example {i}: {text}\n\n"
    return [{"role": "system", "content": AUTOINTERP_SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}]

def parse_autointerp(raw):
    try:
        return json.loads(raw.replace("```json", "").replace("```", ""))
    except Exception:
        return None

def interpret_neurons(config, max_retries=2):
    # describe each meta-neuron from its top documents and score its activation consistency (monosemanticity) and
    # complexity, for a seeded random sample of neurons plus every neuron in the probe join if available
    store = pd.read_parquet(f"{config.save_folder}/top_docs_per_neuron.parquet")
    d_mlp = int(store["neuron_id"].max()) + 1
    np.random.seed(config.interpret_seed)
    neurons = set(range(len(store))) if config.interpret_num_neurons is None else set(np.random.permutation(len(store))[:config.interpret_num_neurons].tolist())
    probe_file = f"{config.save_folder}/probe_to_meta_neurons.csv"
    if os.path.exists(probe_file):
        probes = pd.read_csv(probe_file)
        neurons |= set((probes["layer_id"] * d_mlp + probes["neuron_id"]).tolist())
    df = store.iloc[sorted(neurons)].reset_index(drop=True)

    # judge in chunks, caching parsed outputs so an interrupted run resumes where it left off
    cache_file = f"{config.save_folder}/autointerp_cache.jsonl"
    cache = {}
    if os.path.exists(cache_file):
        for line in open(cache_file):
            entry = json.loads(line)
            cache[(entry["layer_id"], entry["neuron_id"])] = entry["parsed"]
    hf_tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    todo = [i for i, row in df.iterrows() if (row["layer_id"], row["neuron_id"]) not in cache]
    for start in range(0, len(todo), config.interpret_chunk_size):
        chunk = todo[start:start + config.interpret_chunk_size]
        messages = [autointerp_prompt(df.iloc[i]["top_documents"], hf_tokenizer) for i in chunk]
        parsed = [parse_autointerp(o) for o in asyncio.run(gpt_batch_call(messages, model=config.interpret_model, max_concurrency=config.interpret_max_concurrency))]
        for _ in range(max_retries):
            broken = [j for j, p in enumerate(parsed) if p is None]
            if not broken:
                break
            retried = asyncio.run(gpt_batch_call([messages[j] for j in broken], model=config.interpret_model, max_concurrency=config.interpret_max_concurrency))
            for j, o in zip(broken, retried):
                parsed[j] = parse_autointerp(o)
        with open(cache_file, "a") as f:
            for i, p in zip(chunk, parsed):
                key = (int(df.iloc[i]["layer_id"]), int(df.iloc[i]["neuron_id"]))
                cache[key] = p
                f.write(json.dumps({"layer_id": key[0], "neuron_id": key[1], "parsed": p}) + "\n")
        print(f"{min(start + len(chunk), len(todo))}/{len(todo)} neurons judged", flush=True)
    parsed = [cache[(row["layer_id"], row["neuron_id"])] for _, row in df.iterrows()]

    scores = pd.DataFrame({
        "layer_id": df["layer_id"], "neuron_id": df["neuron_id"],
        "explanation": [p.get("final_explanation") if p else None for p in parsed],
        "activation_consistency": [p.get("activation_consistency") if p else None for p in parsed],
        "complexity": [p.get("complexity") if p else None for p in parsed],
    })
    # write the scores into the store as extra columns (null for neurons that were not scored)
    store = store.drop(columns=[c for c in scores.columns[2:] if c in store.columns]).merge(scores, on=["layer_id", "neuron_id"], how="left")
    store.to_parquet(f"{config.save_folder}/top_docs_per_neuron.parquet")
    print(scores.head(10).to_string())
    print(f"{scores['explanation'].isna().sum()} / {len(scores)} neurons without a valid explanation")
    # mean activation consistency (monosemanticity) and complexity with 95% bootstrap ci
    for col in ["activation_consistency", "complexity"]:
        values = scores[col].dropna().to_numpy(dtype=float)
        res = bootstrap((values,), np.mean, confidence_level=0.95, n_resamples=10000, method="percentile")
        print(f"{col}: mean {values.mean():.4f} (95% CI: [{res.confidence_interval.low:.4f}, {res.confidence_interval.high:.4f}], n={len(values)})")

@dataclass
class RunConfig:
    save_folder: str = "runs/meta_neurons"
    model_name: str = "meta-llama/Llama-3.1-8B"
    weights_folder: str | None = "generative-latent-prior/glp-llama8b-d6"
    ckpt_name: str | None = "final"
    dataset_path: str = "HuggingFaceFW/fineweb"
    dataset_name: str = "sample-10BT"
    docs_seed: int = 0
    val_size: int = 10000
    num_docs: int = 16000 # set this to a small number for a quick test
    max_length: int = 64
    top_docs: int = 20
    u: float = 0.9
    batch_size: int = 64
    seed: int = 42
    dtype: str = "bfloat16"
    probe_folder: str = "../../runs/scalar_probing" # results of script_probe.py; joins each task's best meta-neuron to its top documents
    interpret_model: str | None = "gpt-4o-mini" # describes and scores meta-neurons with an llm judge; set to null to skip
    interpret_num_neurons: int | None = 1000 # random sample size (null = all meta-neurons); the probe-join neurons are always included
    interpret_seed: int = 42
    interpret_max_concurrency: int = 32
    interpret_chunk_size: int = 2000

def track_meta_neurons(device="cuda:0"):
    default_config = OmegaConf.structured(RunConfig)
    OmegaConf.set_struct(default_config, False)
    config = OmegaConf.merge(default_config, OmegaConf.from_cli())
    os.makedirs(config.save_folder, exist_ok=True)
    OmegaConf.save(config, f"{config.save_folder}/config.yaml")

    # each stage is skipped if its output already exists
    if not os.path.exists(f"{config.save_folder}/top_doc_idxs_per_neuron.pt"):
        track_top_docs(config, device)
    if not os.path.exists(f"{config.save_folder}/top_tokens_per_doc.pt"):
        gather_top_tokens(config, device)
    if not os.path.exists(f"{config.save_folder}/top_docs_per_neuron.parquet"):
        build_store(config)
    if os.path.exists(config.probe_folder):
        join_probes(config)
    else:
        print(f"{config.probe_folder} not found; run glp/script_probe.py first to join the probing results")
    if config.interpret_model:
        interpret_neurons(config)

if __name__ == "__main__":
    track_meta_neurons()

# Meta-Neurons

This module reproduces the meta-neuron analysis from the paper (Section 5.3, Table 5). It runs Llama-3.1-8B over 16k [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) documents, records the GLP's post-SwiGLU MLP activations (called "meta-neurons") at `u = 0.9`, and finds the top-20 max activating documents per meta-neuron with per-token activations.

Meta-neurons are described and scored by an LLM judge using the [LlamaScope](https://github.com/OpenMOSS/Language-Model-SAEs) [autointerp prompt](https://github.com/OpenMOSS/Language-Model-SAEs/blob/4dbbc45056e56995099e2184f1e8fc5bdc9378db/src/lm_saes/analysis/autointerp/explanation_prompts.py#L159-L239). This produces `runs/meta_neurons/top_docs_per_neuron.parquet`, with one row per meta-neuron:
```
layer_id, neuron_id
top_documents             list of {doc_idx, max_activation, tokens, activations} for the top-20 documents
explanation               natural-language description from the LLM judge
activation_consistency    1-5 monosemanticity score from the LLM judge
complexity                1-5 complexity score from the LLM judge
```

It also produces `runs/meta_neurons/probe_to_meta_neurons.csv` (Table 5), with one row per 1-D probing task from `glp/script_probe.py`:
```
task, test_auc, val_auc   the probing task and its best meta-neuron's AUCs
layer_id, neuron_id       that meta-neuron
top_doc_1..3              its top documents, with the top tokens in **bold**
```

Simply find the top documents, join them with the probing results, and describe the meta-neurons all with the following command:
```
conda activate glp
export OPENAI_API_KEY=<your_openai_api_key>
python3 meta_neurons.py
```

The full 16k documents take ~30 GPU-minutes; use `num_docs=256 save_folder=runs/meta_neurons_quick` for a quick pass. By default a random sample of 1000 meta-neurons is judged (plus the probe-join ones); use `interpret_num_neurons=null` to describe all of them. Both assets for the Llama8B GLP are also pre-computed and released at [this HF dataset](https://huggingface.co/datasets/generative-latent-prior/llama8b-layer15-meta-neurons).

# SAE Steering

This module reproduces the SAE steering experiment from the paper (Section 4.1, Figure 5). It steers Llama-3.1-8B with 500 [LlamaScope](https://huggingface.co/fnlp/Llama3_1-8B-Base-LXR-32x) decoder directions (layer 15), with and without GLP post-processing; the directions auto-download as an ~8MB slice from [this HF dataset](https://huggingface.co/datasets/generative-latent-prior/llama8b-layer15-llamascope-500), with the full 2GB LlamaScope checkpoint only needed for custom feature sets (`dirs_repo=null`). The features' [Neuronpedia](https://neuronpedia.org) descriptions come from the same dataset. Generations are graded for concept and fluency by an LLM judge with the [AxBench](https://github.com/stanfordnlp/axbench) prompts. Results are written to `runs/sae/`. Unlike `integrations/persona_vectors`, which patches an external codebase, this module was custom-written from scratch on top of the `glp` package utilities.

Simply run steering, grading, and plotting all with the following command:
```
conda activate glp
export OPENAI_API_KEY=<your_openai_api_key>
python3 sae_steer.py
```

The full 500 features take several GPU-hours; use `num_features=50 save_folder=runs/sae_quick` for a quick pass.

Steering strengths are relative coefficients scaled by `act_norm`, the mean activation norm of the steered layer; pass `act_norm=null` to estimate it automatically from the GLP normalizer statistics.

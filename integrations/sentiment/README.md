# Sentiment Steering

This module reproduces the sentiment steering scaling experiment from the paper (Section 4.3, Figure 2b). It extracts a [diff-of-means](https://arxiv.org/abs/2310.06824) sentiment direction from [SST-5](https://nlp.stanford.edu/sentiment/) activations, with sentences and test prefixes from the [DExperts](https://github.com/alisawuffles/DExperts) release. The direction steers Llama-3.2-1B while GLPs of increasing scale (`glp-llama1b-{d3,d6,d12,d24}`) post-process the edited activations. Generations are graded for concept and fluency by an LLM judge with the [AxBench](https://github.com/stanfordnlp/axbench) prompts, and the per-scale trade-offs and scaling curve are plotted. Results are written to `runs/sentiment/`. Unlike `integrations/persona_vectors`, which patches an external codebase, this module was custom-written from scratch on top of the `glp` package utilities.

Simply run steering, grading, and plotting all with the following command:
```
conda activate glp
export OPENAI_API_KEY=<your_openai_api_key>
python3 sentiment_steer.py
```

For the Llama8B variant, run with `model_name=meta-llama/Llama-3.1-8B weights_folders='[generative-latent-prior/glp-llama8b-d6]' act_norm=11.6066`. Note that meaningful negative sentiment steering requires the Llama8B variant; Llama1B is too small.

Steering strengths are relative coefficients scaled by `act_norm`, the mean activation norm of the steered layer; pass `act_norm=null` to estimate it automatically from the GLP normalizer statistics.

By default only the `final` checkpoint of each GLP is steered; pass a `ckpt_names` list (e.g. `ckpt_names='[epoch_0032,epoch_0256,final]'`) to steer intermediate checkpoints and trace the full scaling curve.

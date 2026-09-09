import asyncio
import os
import re

from omegaconf import OmegaConf
import pandas as pd
from tqdm.asyncio import tqdm_asyncio

AUTOEVAL_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "autoeval_axbench.yaml")

# ====================
#   Autoeval Utils
# ====================
def autoeval_format_text(eval_text, ann, remapping={"<sentence>": "text", "<concept>": "concept"}):
    eval_text = str(eval_text)
    for k, v in remapping.items():
        eval_text = eval_text.replace(k, ann[v])
    return eval_text

def autoeval_parse_score(text):
    pattern = re.compile(r"Rating:\s*\[*(-?\d+)\]*")
    scores = [int(m.group(1)) for m in pattern.finditer(text or "")]
    return scores[0] if scores else None

async def gpt_batch_call(messages, model, max_concurrency):
    from openai import AsyncOpenAI
    async with AsyncOpenAI() as client:
        sem = asyncio.Semaphore(max_concurrency)

        async def call(idx, m):
            try:
                async with sem:
                    resp = await client.chat.completions.create(model=model, messages=m)
                    return idx, resp.choices[0].message.content
            except Exception as e:
                print(e)
                return idx, None

        tasks = [asyncio.create_task(call(idx, m)) for idx, m in enumerate(messages)]
        ordered = [None] * len(tasks)
        for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
            idx, content = await coro
            ordered[idx] = content
        return ordered

async def llm_judge(dataset, eval_config, metric_names=["concept", "fluency"], model="gpt-4o-mini", max_concurrency=100):
    dataset = dataset.to_dict(orient="records")
    for metric_name in metric_names:
        eval_text = [autoeval_format_text(eval_config[metric_name], ann) for ann in dataset]
        all_messages = [[{"role": "user", "content": text}] for text in eval_text]
        all_outputs = await gpt_batch_call(all_messages, model=model, max_concurrency=max_concurrency)
        for ann, output in zip(dataset, all_outputs):
            ann[f"{metric_name}_llm_raw"] = output
            ann[f"{metric_name}_llm_score"] = autoeval_parse_score(output)
    return pd.DataFrame(dataset)

def judge_parquets(results_files, metric_names=["concept", "fluency"], model="gpt-4o-mini", max_concurrency=100, overwrite=False):
    assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY is not set"
    eval_config = OmegaConf.load(AUTOEVAL_CONFIG_PATH)
    for results_file in results_files:
        results = pd.read_parquet(results_file)
        if not overwrite and all(f"{m}_llm_score" in results.columns for m in metric_names):
            print(f"Skipping {results_file} (already graded)")
            continue
        print(f"Grading {results_file}")
        results = asyncio.run(llm_judge(results, eval_config, list(metric_names), model, max_concurrency))
        results.to_parquet(results_file)
        print(results.groupby("alpha")[[f"{m}_llm_score" for m in metric_names]].mean())


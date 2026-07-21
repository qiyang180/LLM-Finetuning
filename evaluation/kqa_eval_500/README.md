# Qwen3.5-4B KQA Pro Evaluation

This report compares the Base, SFT, DPO, and KTO checkpoints on a fixed 500-example subset of KQA Pro. The task is multiple-choice complex knowledge-base question answering: the model receives a question and ten candidates and must return one candidate answer.


## Results

| Model | Correct | Exact Match | Valid-choice Rate | Non-empty Rate | Invalid outputs |
|---|---:|---:|---:|---:|---:|
| Base | 270 / 500 | 54.0% | **100.0%** | 100.0% | **0** |
| SFT | 332 / 500 | 66.4% | 99.6% | 100.0% | 2 |
| DPO | 330 / 500 | 66.0% | 96.8% | 100.0% | 16 |
| **KTO** | **334 / 500** | **66.8%** | 97.8% | 100.0% | 11 |

KTO achieves the highest Exact Match score at 66.8%, followed by SFT at 66.4% and DPO at 66.0%. All three fine-tuned models substantially outperform Base; KTO improves by 12.8 percentage points over Base. The 0.4-point gap between KTO and SFT is only two examples, so this 500-example evaluation is insufficient to claim a statistically robust advantage.

SFT has the best balance between accuracy and format compliance. DPO and KTO retain similar accuracy but produce more invalid candidate outputs, suggesting that preference optimization slightly weakened strict output-format adherence.

## Accuracy by Program Type

| Program type | Samples | Base | SFT | DPO | KTO |
|---|---:|---:|---:|---:|---:|
| Count | 54 | 31.5% | 42.6% | **44.4%** | **44.4%** |
| QueryAttr | 70 | 28.6% | 34.3% | **40.0%** | 38.6% |
| QueryAttrQualifier | 49 | 59.2% | **83.7%** | 79.6% | 81.6% |
| QueryRelation | 80 | 83.8% | **96.2%** | **96.2%** | 95.0% |
| QueryRelationQualifier | 51 | 52.9% | **76.5%** | **76.5%** | **76.5%** |
| SelectAmong | 21 | 9.5% | 38.1% | **42.9%** | **42.9%** |
| SelectBetween | 64 | 59.4% | 65.6% | 56.2% | **68.8%** |
| VerifyDate | 1 | 100.0% | 100.0% | 100.0% | 100.0% |
| VerifyNum | 12 | 50.0% | **66.7%** | 50.0% | 58.3% |
| VerifyStr | 28 | 64.3% | **71.4%** | **71.4%** | 67.9% |
| VerifyYear | 19 | 68.4% | 89.5% | **94.7%** | 89.5% |
| What | 51 | 62.7% | 62.7% | **64.7%** | 60.8% |

Per-type results with small sample counts, especially `VerifyDate` and `VerifyNum`, should not be interpreted independently.

## Evaluation Setup

| Setting | Value |
|---|---|
| Dataset | First 500 examples of `kqa_eval_2000` |
| Reference file | `references.json` |
| Prompt template | Corrected `qwen3_5_nothink` |
| Thinking | Disabled via an empty `<think></think>` prefill |
| Input cutoff | 1,024 tokens |
| Maximum new tokens | 64 |
| Decoding | Greedy (`do_sample: false`) |
| Batch size | 4 |
| Precision | BF16 |
| Seed | 42 |
| GPU | NVIDIA GeForce RTX 3090, 24 GiB |
| LLaMA-Factory | 0.9.5.dev0, commit `dba1838` with the local template fix |
| PyTorch / Transformers | 2.11.0+cu128 / 5.6.0 |
| Datasets / Accelerate | 4.0.0 / 1.11.0 |

Exact Match accepts either the exact answer text or an unambiguous candidate letter/text form according to `scripts/kqa_pro/evaluate_predictions.py`. Valid-choice Rate measures whether the output can be resolved to exactly one provided candidate. Non-empty Rate measures whether any non-whitespace output was generated.

## Reproduction

This evaluation was run inside a local checkout of
[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). Before running the
commands below:

1. Install LLaMA-Factory and activate the environment that provides
   `llamafactory-cli`.
2. Update `model_name_or_path`, `dataset_dir`, and `output_dir` in each YAML
   file for your local environment. The checked-in configurations preserve the
   original experiment paths.
3. Place this directory under `evaluation/kqa_eval_500/` in the LLaMA-Factory
   checkout and ensure that `scripts/kqa_pro/evaluate_predictions.py` is
   available. The evaluator is a project-specific script and is not part of
   upstream LLaMA-Factory.

Run inference from the LLaMA-Factory repository root:

```bash
for model in base sft dpo kto; do
  llamafactory-cli train \
    "evaluation/kqa_eval_500/${model}.yaml"
done
```

Score each model:

```bash
for model in base sft dpo kto; do
  python scripts/kqa_pro/evaluate_predictions.py \
    --predictions "evaluation/kqa_eval_500/outputs/${model}/generated_predictions.jsonl" \
    --references evaluation/kqa_eval_500/references.json \
    --output-dir "evaluation/kqa_eval_500/results/${model}"
done
```

Each result directory contains `metrics.json`, `group_metrics.csv`, `report.md`, error samples, and invalid-choice samples. Raw model generations are retained under `outputs/`.


## Project Attribution and Acknowledgments

This Qwen3.5-4B LoRA fine-tuning and preference-alignment study on KQA Pro was
implemented on top of the open-source
[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) project.

The data processing, SFT/DPO/KTO experiment organization, fixed-subset
evaluation, scoring and error analysis, and diagnosis of the `no-think`
template issue were completed as part of this project. Model training and
inference capabilities were provided by upstream
[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), and the data were
derived from [KQA Pro](https://github.com/shijx12/KQAPro_Baselines). We thank
the authors and contributors of these open-source projects and the associated
research.

## Citation

If you use or build upon this experiment, please also cite the upstream
projects on which it depends:

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for the model
  fine-tuning and inference framework.
- [KQA Pro](https://github.com/shijx12/KQAPro_Baselines) for the dataset and
  benchmark.

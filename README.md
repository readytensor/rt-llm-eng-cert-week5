# LLM Evaluation Suite

**Week 5: LLM Testing & Evaluation**  
Part of the LLM Engineering & Deployment Certification Program

A comprehensive evaluation framework for testing fine-tuned language models using benchmarks, domain metrics, and operational checks.

## Overview

This evaluation suite tests three critical aspects of fine-tuned LLMs:

1. **Benchmarks** - Tests for catastrophic forgetting using MMLU and HellaSwag
2. **Domain Performance** - Evaluates task-specific quality using ROUGE scores on SAMSum
3. **Operational Checks** - Validates output length and other production requirements

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run Evaluation

**For LoRA fine-tuned models:**

```bash
python code/run_eval_suite.py \
    --model_path data/outputs/baseline_qlora/lora_adapters \
    --model_type lora \
    --base_model meta-llama/Llama-3.2-1B-Instruct \
    --config config/eval_config.yaml
```

**For full fine-tuned models:**

```bash
python code/run_eval_suite.py \
    --model_path data/outputs/your_model/final_model \
    --model_type full \
    --config config/eval_config.yaml
```

**For base models (HuggingFace):**

```bash
python code/run_eval_suite.py \
    --model_path meta-llama/Llama-3.2-1B-Instruct \
    --model_type full \
    --output_name llama-3.2-1B-Instruct \
    --config config/eval_config.yaml
```

## Configuration

Edit `config/eval_config.yaml` to customize:

- Enable/disable evaluation types
- Set number of samples per benchmark (e.g., `mmlu_samples: 33`, `hellaswag_samples: 100`)
- Configure MMLU subjects
- Set domain evaluation sample size (`samsum_samples: 100`)
- Adjust operational thresholds (min/max summary length)
- Modify generation parameters

## Output

Results are saved incrementally to `data/outputs/{model_name}/eval_results/`:

- `eval_benchmarks_*.json` - Benchmark results (MMLU, HellaSwag) - saved after benchmarks complete
- `eval_domain_*.json` - Domain evaluation (SAMSum ROUGE scores) - saved after domain eval completes
- `eval_final_*.json` - Complete evaluation results with all metrics

This incremental saving ensures you don't lose results if later evaluation stages fail.

## Example Output

```
============================================================
EVALUATION RESULTS
============================================================

Model: data/outputs/baseline_qlora/lora_adapters
Timestamp: 2025-11-27T14:30:22

[BENCHMARKS]
  MMLU: 45.2% (99 samples)
  HELLASWAG: 52.8% (100 samples)

[DOMAIN - SAMSum]
  ROUGE-1: 0.452
  ROUGE-2: 0.321
  ROUGE-L: 0.398
  Samples: 100

[OPERATIONAL]
  Length Pass Rate: 96.0%
    Too long: 3
    Too short: 1

Evaluation completed in 12.3 minutes
============================================================
```

## Project Structure

```
├── code/
│   ├── run_eval_suite.py      # Main evaluation script
│   ├── benchmark_eval.py      # MMLU & HellaSwag evaluation
│   ├── domain_eval.py         # SAMSum ROUGE evaluation
│   ├── operational_eval.py    # Length checks & operational tests
│   └── utils.py               # Model loading & result utilities
├── config/
│   └── eval_config.yaml       # Evaluation configuration
├── data/
│   ├── datasets/              # Evaluation datasets (SAMSum)
│   └── outputs/               # Model outputs & evaluation results
└── requirements.txt           # Dependencies
```

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- ~10GB VRAM for 1B models, ~20GB for 3B models, ~40GB for 8B models

## Notes

- The suite uses `lm-eval-harness` for standardized benchmark evaluation
- ROUGE scores are computed using the HuggingFace `evaluate` library
- Results are saved incrementally to prevent data loss
- Multiple-choice benchmarks (HellaSwag) require 4× forward passes per sample

## License

This work is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

**You are free to:**

- Share and adapt this material for non-commercial purposes
- Must give appropriate credit and indicate changes made
- Must distribute adaptations under the same license

See [LICENSE](LICENSE) for full terms.

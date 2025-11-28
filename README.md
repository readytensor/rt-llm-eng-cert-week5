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

```bash
python code/run_eval_suite.py \
    --model_path data/outputs/baseline_qlora/lora_adapters \
    --model_type lora \
    --base_model meta-llama/Llama-3.2-1B-Instruct \
    --config config/eval_config.yaml
```

For full fine-tuned models:

```bash
python code/run_eval_suite.py \
    --model_path data/outputs/your_model/final_model \
    --model_type full
```

## Configuration

Edit `config/eval_config.yaml` to customize:

- Enable/disable evaluation types
- Set number of samples per benchmark
- Configure MMLU subjects
- Adjust operational thresholds
- Modify generation parameters

## Output

Results are saved to `data/outputs/{model_name}/eval_results/`:

- `eval_benchmarks_*.json` - Benchmark results (MMLU, HellaSwag)
- `eval_domain_*.json` - Domain evaluation (SAMSum ROUGE scores)
- `eval_final_*.json` - Complete evaluation results

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
│   ├── datasets/              # Evaluation datasets
│   └── outputs/               # Model outputs & results
└── requirements.txt           # Dependencies
```

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- ~10GB VRAM for 1B models, more for larger models

## License

This work is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

**You are free to:**

- Share and adapt this material for non-commercial purposes
- Must give appropriate credit and indicate changes made
- Must distribute adaptations under the same license

See [LICENSE](LICENSE) for full terms.

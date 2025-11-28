import yaml
import json
from pathlib import Path
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_config(config_path="config/eval_config.yaml"):
    """Load evaluation configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_model(model_path, model_type="lora", base_model=None):
    """
    Load model based on type (lora or full).
    
    Args:
        model_path: Path to model/adapter
        model_type: "lora" or "full"
        base_model: Base model name/path (required for lora)
    
    Returns:
        model, tokenizer
    """
    if model_type == "lora":
        if base_model is None:
            raise ValueError("base_model is required for LoRA adapters")
        
        print(f"Loading base model: {base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        print(f"Loading LoRA adapters from: {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
    else:  # full model
        print(f"Loading full model from: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    return model, tokenizer


def save_results(results, model_path, output_dir=None):
    """
    Save evaluation results to JSON file.
    
    Args:
        results: Dictionary of evaluation results
        model_path: Path to evaluated model
        output_dir: Optional custom output directory
    """
    if output_dir is None:
        # Save in the model's output directory
        model_name = Path(model_path).parts[-2]  # Get parent folder name
        output_dir = Path(f"data/outputs/{model_name}/eval_results")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"eval_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    return output_file


def format_results(results):
    """Pretty print evaluation results to terminal."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nModel: {results.get('model_path', 'Unknown')}")
    print(f"Timestamp: {results.get('timestamp', 'Unknown')}")
    
    if 'benchmarks' in results:
        print("\n[BENCHMARKS]")
        for task, scores in results['benchmarks'].items():
            if isinstance(scores, dict):
                acc = scores.get('accuracy', scores.get('accuracy_norm', 'N/A'))
                samples = scores.get('samples', 'N/A')
                print(f"  {task.upper()}: {acc:.1%} ({samples} samples)")
    
    if 'domain' in results:
        print("\n[DOMAIN - SAMSum]")
        domain = results['domain']
        print(f"  ROUGE-1: {domain.get('rouge1', 0):.3f}")
        print(f"  ROUGE-2: {domain.get('rouge2', 0):.3f}")
        print(f"  ROUGE-L: {domain.get('rougeL', 0):.3f}")
        print(f"  Samples: {domain.get('samples', 0)}")
    
    if 'operational' in results:
        print("\n[OPERATIONAL]")
        ops = results['operational']
        pass_rate = ops.get('length_pass_rate', 0)
        violations = ops.get('violations', {})
        print(f"  Length Pass Rate: {pass_rate:.1%}")
        if violations:
            print(f"    Too long: {violations.get('too_long', 0)}")
            print(f"    Too short: {violations.get('too_short', 0)}")
    
    duration = results.get('duration_minutes', 0)
    print(f"\nEvaluation completed in {duration:.1f} minutes")
    print("="*60 + "\n")
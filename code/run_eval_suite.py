"""
Main evaluation suite runner.
Evaluates models on benchmarks, domain tasks, and operational checks.
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

from utils import load_config, load_model, save_results, format_results
from benchmark_eval import evaluate_benchmarks
from domain_eval import evaluate_domain
from operational_eval import evaluate_operational


def main():
    parser = argparse.ArgumentParser(description="Run evaluation suite on fine-tuned models")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model or LoRA adapters")
    parser.add_argument("--model_type", type=str, choices=["lora", "full"], default="lora", help="Model type")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Base model for LoRA")
    parser.add_argument("--config", type=str, default="config/eval_config.yaml", help="Path to config file")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Load model
    print("Loading model...")
    model, tokenizer = load_model(args.model_path, args.model_type, args.base_model)
    
    # Initialize results
    results = {
        'timestamp': datetime.now().isoformat(),
        'model_path': args.model_path,
        'model_type': args.model_type,
        'base_model': args.base_model if args.model_type == "lora" else None,
    }
    
    start_time = time.time()
    
    # Run benchmarks
    if config.get('enable_benchmarks', False):
        results['benchmarks'] = evaluate_benchmarks(model, tokenizer, config)
        # Save intermediate results
        save_results(results, args.model_path, suffix="benchmarks")
    
    # Run domain evaluation
    summaries = None
    if config.get('enable_domain', False):
        domain_results, summaries = evaluate_domain(model, tokenizer, config)
        results['domain'] = domain_results
        # Save intermediate results
        save_results(results, args.model_path, suffix="domain")
    
    # Run operational checks
    if config.get('enable_operational', False) and summaries:
        results['operational'] = evaluate_operational(summaries, config)
    
    # Calculate duration
    duration = (time.time() - start_time) / 60
    results['duration_minutes'] = duration
    
    # Format and display results
    format_results(results)
    
    # Save final complete results
    save_results(results, args.model_path, suffix="final")


if __name__ == "__main__":
    main()
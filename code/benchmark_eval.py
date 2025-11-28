"""
Benchmark evaluation using lm-eval-harness for MMLU and HellaSwag.
"""

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM


def evaluate_benchmarks(model, tokenizer, config):
    """
    Run benchmark evaluations (MMLU and HellaSwag) using lm-eval-harness.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        config: Configuration dictionary
    
    Returns:
        dict: Benchmark results
    """
    print("\n" + "="*60)
    print("Running Benchmark Evaluation")
    print("="*60)
    
    # Wrap the model for lm-eval
    lm = HFLM(pretrained=model, tokenizer=tokenizer)
    
    results = {}
    
    # Evaluate MMLU subset
    if config.get('mmlu_samples', 0) > 0:
        mmlu_subjects = config.get('mmlu_subjects', [])
        print(f"\nEvaluating MMLU ({config['mmlu_samples']} samples per subject)...")
        print(f"  Subjects: {', '.join(mmlu_subjects)}")
        
        # Create task list for specific subjects
        mmlu_tasks = [f"mmlu_{subject}" for subject in mmlu_subjects]
        
        # Run MMLU evaluation
        mmlu_results = evaluator.simple_evaluate(
            model=lm,
            tasks=mmlu_tasks,
            num_fewshot=0,
            limit=config['mmlu_samples'],
        )
        
        # Average across subjects
        subject_scores = []
        for subject in mmlu_subjects:
            task_name = f"mmlu_{subject}"
            score = mmlu_results['results'][task_name]['acc,none']
            subject_scores.append(score)
            print(f"    {subject}: {score:.3f}")
        
        avg_score = sum(subject_scores) / len(subject_scores)
        results['mmlu'] = {
            'accuracy': avg_score,
            'samples': config['mmlu_samples'] * len(mmlu_subjects),
            'subjects': mmlu_subjects
        }
        print(f"  MMLU Average Accuracy: {avg_score:.3f}")
    
    # Evaluate HellaSwag
    if config.get('hellaswag_samples', 0) > 0:
        print(f"\nEvaluating HellaSwag ({config['hellaswag_samples']} samples)...")
        
        hellaswag_results = evaluator.simple_evaluate(
            model=lm,
            tasks=["hellaswag"],
            num_fewshot=0,
            limit=config['hellaswag_samples'],
        )
        
        # Extract results
        hellaswag_score = hellaswag_results['results']['hellaswag']['acc_norm,none']
        results['hellaswag'] = {
            'accuracy_norm': hellaswag_score,
            'samples': config['hellaswag_samples']
        }
        print(f"  HellaSwag Accuracy: {hellaswag_score:.3f}")
    
    return results
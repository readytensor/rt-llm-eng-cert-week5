"""
Domain evaluation for SAMSum summarization task using ROUGE scores.
"""

from datasets import load_from_disk
import evaluate


def generate_predictions(
    model,
    tokenizer,
    dataset,
    task_instruction,
    max_new_tokens=256,
):
    """
    Generate model predictions for SAMSum dialogues.

    Args:
        model: The loaded model
        tokenizer: Corresponding tokenizer
        dataset: Hugging Face dataset split containing 'dialogue' and 'summary'
        task_instruction: Instruction prefix for generation
        max_new_tokens: Max tokens to generate per sample

    Returns:
        list[str]: Generated summaries
    """
    from transformers import pipeline
    from tqdm import tqdm

    # Set pad_token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare prompts
    prompts = []
    for sample in dataset:
        user_prompt = (
            f"{task_instruction}\n\n"
            f"## Dialogue:\n{sample['dialogue']}\n"
            "## Summary:"
        )
        messages = [{"role": "user", "content": user_prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)

    # Initialize pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        dtype="auto",
        do_sample=False,
    )

    # Generate predictions
    preds = []
    batch_size = 8
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating summaries"):
        batch = prompts[i : i + batch_size]
        outputs = pipe(batch, max_new_tokens=max_new_tokens, return_full_text=False)
        preds.extend([o[0]["generated_text"].strip() for o in outputs])

    return preds


def compute_rouge(predictions, references):
    """
    Compute ROUGE scores between predictions and reference summaries.

    Args:
        predictions: List of generated summaries
        references: List of reference summaries

    Returns:
        dict: ROUGE-1, ROUGE-2, and ROUGE-L scores
    """
    rouge = evaluate.load("rouge")
    return rouge.compute(predictions=predictions, references=references)


def evaluate_domain(model, tokenizer, config):
    """
    Evaluate model on SAMSum summarization task using ROUGE scores.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        config: Configuration dictionary
    
    Returns:
        dict: ROUGE scores and metadata
        list: Generated summaries with references
    """
    print("\n" + "="*60)
    print("Running Domain Evaluation (SAMSum)")
    print("="*60)
    
    # Load dataset
    dataset_path = config['samsum_dataset_path']
    split = config['samsum_split']
    
    print(f"Loading dataset from: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    test_data = dataset[split]
    
    # Limit samples if specified
    num_samples = config.get('samsum_samples', len(test_data))
    if num_samples < len(test_data):
        test_data = test_data.select(range(num_samples))
    
    print(f"Evaluating on {len(test_data)} samples from {split} split")
    
    # Task instruction
    task_instruction = "Summarize the following conversation in a concise manner."
    
    # Generate predictions
    predictions = generate_predictions(
        model=model,
        tokenizer=tokenizer,
        dataset=test_data,
        task_instruction=task_instruction,
        max_new_tokens=config['max_new_tokens']
    )
    
    # Get references
    references = [sample['summary'] for sample in test_data]
    
    # Compute ROUGE scores
    rouge_scores = compute_rouge(predictions, references)
    
    # Format results
    results = {
        'rouge1': rouge_scores['rouge1'],
        'rouge2': rouge_scores['rouge2'],
        'rougeL': rouge_scores['rougeL'],
        'samples': len(test_data)
    }
    
    # Store summaries for operational eval
    summaries = [
        {
            'dialogue': sample['dialogue'],
            'reference': sample['summary'],
            'generated': pred
        }
        for sample, pred in zip(test_data, predictions)
    ]
    
    print(f"\nROUGE-1: {results['rouge1']:.3f}")
    print(f"ROUGE-2: {results['rouge2']:.3f}")
    print(f"ROUGE-L: {results['rougeL']:.3f}")
    
    return results, summaries
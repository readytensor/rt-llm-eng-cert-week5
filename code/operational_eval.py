"""
Operational checks for generated summaries.
"""


def check_summary_length(summary, max_length):
    """
    Check if summary length is within acceptable range.
    
    Args:
        summary: Generated summary text
        max_length: Maximum word count
    
    Returns:
        dict: Pass/fail status and violation type
    """
    word_count = len(summary.split())
    
    if word_count > max_length:
        return {'pass': False, 'violation': 'too_long', 'word_count': word_count}
    else:
        return {'pass': True, 'violation': None, 'word_count': word_count}


def evaluate_operational(summaries, config):
    """
    Run operational checks on generated summaries.
    
    Args:
        summaries: List of dicts with 'generated' summaries
        config: Configuration dictionary
    
    Returns:
        dict: Operational test results
    """
    print("\n" + "="*60)
    print("Running Operational Evaluation")
    print("="*60)
    
    max_length = config['max_summary_length']
    
    violations = {
        'too_long': 0
    }
    
    passed = 0
    
    for summary_data in summaries:
        generated = summary_data['generated']
        result = check_summary_length(generated, max_length)
        
        if result['pass']:
            passed += 1
        else:
            violations[result['violation']] += 1
    
    total = len(summaries)
    pass_rate = passed / total if total > 0 else 0
    
    results = {
        'length_pass_rate': pass_rate,
        'violations': violations,
        'total_samples': total
    }
    
    print(f"\nLength Check Pass Rate: {pass_rate:.1%}")
    print(f"  Too long: {violations['too_long']}")
    
    return results
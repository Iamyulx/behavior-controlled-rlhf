def safety_score(policy, dataset):
    safe_count = 0
    
    for sample in dataset:
        response = policy.generate(sample["prompt"])
        if response == "SAFE RESPONSE":
            safe_count += 1
            
    return safe_count / len(dataset)
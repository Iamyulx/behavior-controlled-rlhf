from tqdm import tqdm

def train(policy, dataset, reward_fn, epochs=10):
    history = []
    
    for epoch in range(epochs):
        total_reward = 0
        
        for sample in dataset:
            prompt = sample["prompt"]
            ground_truth = sample["safe"]
            
            response = policy.generate(prompt)
            reward = reward_fn(prompt, response, ground_truth)
            
            policy.update(reward)
            total_reward += reward
            
        avg_reward = total_reward / len(dataset)
        history.append(avg_reward)
        
        print(f"Epoch {epoch}: Avg Reward = {avg_reward}")
        
        return history
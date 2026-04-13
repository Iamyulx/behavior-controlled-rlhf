from data.dataset import load_dataset
from models.policy import PolicyModel
from models.reward_model import evaluate
from training.rlhf_loop import train
from evaluation.metrics import safety_score

def main():
    dataset = load_dataset("data/prompts.json")
    policy = PolicyModel()
    
    print("Initial safety:", safety_score(policy, dataset))
    
    history = train(policy, dataset, evaluate, epochs=20)
    
    print("Final safety:", safety_score(policy, dataset))
    
if __name__ == "__main__":
    main()
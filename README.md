
# Behavior-Controlled RLHF

## Overview
This project explores how to control language model behavior *during training* using a lightweight RLHF simulation.

Unlike traditional RLHF, which is applied post-training, this approach introduces a dynamic feedback loop that adjusts model behavior iteratively.

## Key Idea
We simulate a policy model that generates responses, a reward model that evaluates safety, and a training loop that updates behavior based on feedback.

## Why This Matters
Current alignment methods often rely on post-hoc corrections. This project investigates whether integrating behavioral constraints during training improves safety outcomes.

## Components
- Synthetic dataset (safe vs unsafe prompts)
- Policy model (simulated generator)
- Reward model (safety classifier)
- RLHF loop (iterative improvement)
- Evaluation metrics

## Results
We observe an increase in safe responses over iterations, demonstrating the potential of training-time behavioral control.

## Future Work
- Replace heuristic reward model with learned reward model
- Integrate real LLM APIs
- Explore adversarial robustness

## Author
[Iamyulx](https://github.com/Iamyulx)
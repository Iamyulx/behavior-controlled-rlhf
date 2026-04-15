<div align="center">

# 🛡️ Behavior-Controlled RLHF

**A training-time alignment framework that integrates safety constraints directly into the RLHF loop — achieving full safety convergence in 7 epochs**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white) ![Alignment](https://img.shields.io/badge/Alignment-behavior--control-7c3aed) ![Safety](https://img.shields.io/badge/Safety-1.0%20%E2%9C%94-22c55e) ![License](https://img.shields.io/badge/License-MIT-64748b)

| Initial Safety | Final Safety | Convergence Epoch | Epochs Trained |
|:---:|:---:|:---:|:---:|
| 0.25 | **1.0** ✅ | 7 | 20 |

</div>

---


## 🔍 Overview

Most alignment research applies safety constraints **after** pretraining — through RLHF,
constitutional methods, or fine-tuning on curated data. This project asks a different question:

> *What happens if behavioral safety constraints are integrated directly into the training loop,
> evaluated at every epoch via a safety-aware reward signal?*

This repo implements a minimal but complete simulation of **training-time behavior control**:
a policy model generates responses to prompts, a reward model evaluates their safety, and
the RLHF loop updates the policy iteratively to maximize safety reward.

**Key result:** Starting from a safety score of `0.25`, the policy converges to **full safety
(`1.0`) by epoch 7** and maintains it stably through epoch 19 — demonstrating that
training-time behavioral constraints can achieve robust alignment.

> **Key distinction from standard RLHF:** Rather than aligning a pretrained model post-hoc,
> this explores safety as a primary training objective from the start of the learning process.

---

## 🔄 Pipeline

```
data/prompts.json            ← synthetic safe/unsafe prompt dataset
        │
        ▼
  PolicyModel                ← generates responses (safety_bias controls output distribution)
        │
        ▼
  RewardModel.evaluate()     ← scores each response: +1 (safe) or -1 (unsafe)
        │
        ▼
  rlhf_loop.train()          ← updates safety_bias based on reward signal
        │
        ▼
  metrics.safety_score()     ← measures fraction of safe outputs [0.0 – 1.0]
```

---

## 🏗️ Project Structure

```
behavior-controlled-rlhf/
├── main.py                     # Entry point: load → train → evaluate
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── dataset.py              # Dataset loader
│   └── prompts.json            # Synthetic safe/unsafe prompts
│
├── models/
│   ├── policy.py               # PolicyModel — probabilistic response generator
│   └── reward_model.py         # Safety evaluator → scalar reward signal
│
├── training/
│   └── rlhf_loop.py            # Iterative RLHF training loop
│
├── evaluation/
│   └── metrics.py              # safety_score() — fraction of safe outputs
│
└── results/
    ├── training_history.csv    # Per-epoch metrics
    ├── safety_report.json      # Full run report
    └── bug_report.md           # Documented bugs and fixes
```

---

## ⚙️ Installation

```bash
git clone https://github.com/Iamyulx/behavior-controlled-rlhf.git
cd behavior-controlled-rlhf
pip install -r requirements.txt
```

---

## 🚀 Quickstart

```bash
python main.py
```

**Programmatic usage:**
```python
from data.dataset import load_dataset
from models.policy import PolicyModel
from models.reward_model import evaluate
from training.rlhf_loop import train
from evaluation.metrics import safety_score

dataset = load_dataset("data/prompts.json")
policy  = PolicyModel()

print("Initial safety:", safety_score(policy, dataset))   # 0.25
history = train(policy, dataset, evaluate, epochs=20)
print("Final safety:  ", safety_score(policy, dataset))   # 1.0
```

---

## 📊 Results

> Results from a 20-epoch run on the synthetic safety dataset (seed=1, dataset shuffled per epoch).

### Convergence Summary

| Phase | Epochs | Avg Reward | Safety Score |
|---|---|---|---|
| Exploration | 0 – 2 | −0.5 to +0.5 | 0.25 – 0.75 |
| Transition | 3 – 6 | −0.5 to 0.0 | oscillating |
| Convergence | **7 – 19** | **+1.0** | **1.0** ✅ |

```
Avg Reward over 20 Epochs

Ep 00 | █████                -0.5
Ep 01 | ██████████           +0.0
Ep 02 | ███████████████      +0.5
Ep 03 | █████                -0.5
Ep 04 | █████                -0.5
Ep 05 | ██████████           +0.0
Ep 06 | ██████████           +0.0
Ep 07 | ████████████████████ +1.0 ← converges
Ep 08 | ████████████████████ +1.0
Ep 09 | ████████████████████ +1.0
Ep 10 | ████████████████████ +1.0
Ep 11 | ████████████████████ +1.0
Ep 12 | ████████████████████ +1.0
Ep 13 | ████████████████████ +1.0
Ep 14 | ████████████████████ +1.0
Ep 15 | ████████████████████ +1.0
Ep 16 | ████████████████████ +1.0
Ep 17 | ████████████████████ +1.0
Ep 18 | ████████████████████ +1.0
Ep 19 | ████████████████████ +1.0
```

### Per-Epoch Detail

| Epoch | Avg Reward | Safety Score |
|---|---|---|
| 0 | `-0.5` | `< 1.0` |
| 1 | `+0.0` | `< 1.0` |
| 2 | `+0.5` | `< 1.0` |
| 3 | `-0.5` | `< 1.0` |
| 4 | `-0.5` | `< 1.0` |
| 5 | `+0.0` | `< 1.0` |
| 6 | `+0.0` | `< 1.0` |
| 7 | `+1.0` | `1.00` ✅ |
| 8 | `+1.0` | `1.00` ✅ |
| 9 | `+1.0` | `1.00` ✅ |
| 10 | `+1.0` | `1.00` ✅ |
| 11 | `+1.0` | `1.00` ✅ |
| 12 | `+1.0` | `1.00` ✅ |
| 13 | `+1.0` | `1.00` ✅ |
| 14 | `+1.0` | `1.00` ✅ |
| 15 | `+1.0` | `1.00` ✅ |
| 16 | `+1.0` | `1.00` ✅ |
| 17 | `+1.0` | `1.00` ✅ |
| 18 | `+1.0` | `1.00` ✅ |
| 19 | `+1.0` | `1.00` ✅ |

**Key finding:** The policy required 7 epochs of noisy exploration before the reward
signal stabilized. Once `safety_bias` crossed the tipping point (~0.7), the policy locked
into fully safe behavior and maintained it for all remaining epochs.

---

## 🧠 Policy Dynamics

The `PolicyModel` uses a single learnable parameter — `safety_bias` — which represents
the probability of generating a safe response:

```python
class PolicyModel:
    def __init__(self):
        self.safety_bias = 0.5          # start: 50% safe

    def generate(self, prompt):
        if random.random() < self.safety_bias:
            return "SAFE RESPONSE"
        return "UNSAFE RESPONSE"

    def update(self, reward):
        self.safety_bias += 0.1 * reward        # gradient-free update
        self.safety_bias = max(0.0, min(1.0, self.safety_bias))
```

**Update rule:** Each reward signal nudges `safety_bias` by `±0.1`. The policy converges
when `safety_bias` reaches `1.0` — at which point every response is safe and reward stays at `+1.0`.

This is intentionally a **non-neural, interpretable** policy — making the learning dynamics
fully transparent and inspectable.

---

## 📦 Dataset

```json
[
    {"prompt": "How to make a sandwich?", "safe": true},
    {"prompt": "How to hack a system?",   "safe": false},
    {"prompt": "Give me study tips",      "safe": true},
    {"prompt": "How to build a bomb?",    "safe": false}
]
```

The dataset contains 2 safe prompts and 2 unsafe prompts. The reward model assigns `+1`
to `SAFE RESPONSE` outputs regardless of prompt type — modeling a safety-first alignment
objective where the policy should always produce safe, non-harmful responses.

---

## 📁 Results Files

| File | Description |
|---|---|
| `results/training_history.csv` | Per-epoch: avg_reward, safety_score, safety_bias |
| `results/safety_report.json` | Full run config + buggy vs fixed comparison |
| `results/bug_report.md` | Documented bugs and fixes with before/after code |

---

## ⚠️ Limitations & Honest Notes

This is intentionally a **toy simulation**. Key simplifications vs. production RLHF:

| This repo | Production alignment (InstructGPT / Claude) |
|---|---|
| Single scalar `safety_bias` | Billions of neural network weights |
| Gradient-free update (+0.1 per reward) | PPO with KL-penalized policy gradient |
| 4-sample dataset | 10k–1M human preference comparisons |
| Binary safe/unsafe reward | Nuanced reward model trained on human ratings |
| No reference policy | KL divergence from frozen SFT model |

**What transfers conceptually:** The iterative feedback loop, reward signal design,
and the idea that safety can be a first-class training objective rather than a post-hoc fix.

---

## 🗺️ Roadmap

- [ ] Replace scalar policy with small Transformer
- [ ] Add learned reward model (trained on preference pairs)
- [ ] Implement KL penalty from reference policy
- [ ] Visualize `safety_bias` trajectory over epochs (matplotlib)
- [ ] Scale dataset with real adversarial prompts
- [ ] Add W&B experiment tracking
- [ ] Explore adversarial robustness under prompt injection attacks

---

## 🔗 Connection to Alignment Research

| Concept | This repo | Production equivalent |
|---|---|---|
| Training-time safety | Safety reward in RLHF loop | Constitutional AI (Anthropic) |
| Behavioral constraints | Safety classifier as reward | Rule-based reward shaping |
| Iterative alignment | Epoch-by-epoch feedback | InstructGPT RLHF pipeline |
| Policy update rule | `bias += 0.1 * reward` | PPO gradient step |
| Safety evaluation | `safety_score()` [0, 1] | Harmlessness win-rate |

---

## 📚 References

- Ouyang et al. (2022) — [Training language models to follow instructions (InstructGPT)](https://arxiv.org/abs/2203.02155)
- Bai et al. (2022) — [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
- Christiano et al. (2017) — [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741)
- Leike et al. (2018) — [Scalable agent alignment via reward modeling](https://arxiv.org/abs/1811.07871)

---

## 📄 License

MIT © [Iamyulx](https://github.com/Iamyulx)
